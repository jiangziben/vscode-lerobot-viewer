// DatasetService
//
// Single source of truth for the set of datasets currently visible in the
// extension. Owns:
//   - Workspace scanning (depth-limited, skips heavy folders)
//   - Manually-added folders (persisted in globalState)
//   - Hugging Face descriptors (metadata cached on disk)
//   - On-demand DatasetSnapshot loading with a tiny LRU
//
// Emits onDidChange whenever the descriptor list mutates so the
// TreeDataProvider can refresh.

import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as vscode from "vscode";
import { log, logError } from "../log";
import type { DatasetDescriptor, DatasetSnapshot, LeRobotEpisode, SshTarget } from "../types";
import { ensureHuggingFaceDataset } from "./huggingface";
import { isLeRobotDataset, loadDataset } from "./datasetLoader";
import { detectDatasetVersion } from "./DatasetVersionDetector";
import { writeStatsJsonl } from "./statsJson";
import { buildParquetSchema } from "./parquetSchema";
import { getAdapter } from "./adapters";
import type { TaskInfo } from "../types";
import {
  fetchSshDataset,
  setPinnedTargets,
  sshCacheDir,
  sshCacheRoot,
  SSH_CACHE_LAST_ACCESS,
} from "./ssh";
import { buildDataPath, buildVideoPath, exists } from "./adapters/util";

const STATE_KEY = "lerobotViewer.descriptors";

// SSH cache dirs untouched for this long get wiped on activate, even
// if their dataset is still registered. The descriptor stays in the
// tree; the next open just re-downloads meta. 1 day matches the
// user's expectation of "if I haven't used it today, I don't need
// the cache lying around."
const STALE_CACHE_THRESHOLD_MS = 24 * 60 * 60 * 1000;
const SKIP_DIR_NAMES = new Set([
  "node_modules",
  ".git",
  ".venv",
  "venv",
  "__pycache__",
  ".cache",
  "dist",
  "build",
  "out",
]);

export class DatasetService implements vscode.Disposable {
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChange = this._onDidChange.event;

  private descriptors: DatasetDescriptor[] = [];
  private readonly snapshotCache = new Map<string, DatasetSnapshot>();
  private readonly loadingPromises = new Map<string, Promise<DatasetSnapshot>>();

  constructor(private readonly context: vscode.ExtensionContext) {
    this.descriptors = this.readPersisted();
    this.refreshSshPins();
    // Background cache maintenance, both fire-and-forget. Errors are
    // never fatal — caches just won't be cleaned up this session.
    void this.cleanOrphanSshCaches();
    void this.cleanStaleSshCaches();
  }

  list(): DatasetDescriptor[] {
    return this.descriptors.slice();
  }

  get(id: string): DatasetDescriptor | undefined {
    return this.descriptors.find((d) => d.id === id);
  }

  /**
   * Ensure a snapshot is loaded for the given descriptor. Concurrent calls
   * for the same id share a single in-flight promise.
   */
  async getSnapshot(id: string): Promise<DatasetSnapshot> {
    const cached = this.snapshotCache.get(id);
    if (cached) return cached;
    const inflight = this.loadingPromises.get(id);
    if (inflight) return inflight;

    const descriptor = this.get(id);
    if (!descriptor) throw new Error(`Unknown dataset id: ${id}`);

    const promise = (async () => {
      const snap = await loadDataset(descriptor);
      this.snapshotCache.set(id, snap);
      this.loadingPromises.delete(id);
      return snap;
    })();
    this.loadingPromises.set(id, promise);
    return promise;
  }

  invalidate(id?: string): void {
    if (id) {
      this.snapshotCache.delete(id);
    } else {
      this.snapshotCache.clear();
    }
  }

  /**
   * Add every LeRobot dataset found under `uri`. The picked folder is
   * itself checked first, then a bounded BFS recurses inside (including
   * past nested datasets, while pruning their data/videos/meta
   * chunks). Datasets already registered are silently skipped; a
   * folder containing no datasets is also a no-op (no error toast),
   * matching the "just figure it out" UX. Returns the descriptors that
   * were freshly added.
   */
  async addLocalFolder(uri: vscode.Uri): Promise<DatasetDescriptor[]> {
    const found = await vscode.window.withProgress<string[]>(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Scanning ${uri.fsPath} for LeRobot datasets`,
        cancellable: true,
      },
      (progress, token) =>
        findLocalDatasets(uri.fsPath, token, (state) =>
          progress.report({
            message: `found ${state.found} · scanned ${state.scanned} · ${shortenLocalProgress(state.currentDir, uri.fsPath)}`,
          }),
        ),
    );

    // Dedupe by root path, not just id, so a folder discovered by
    // workspace auto-scan (id "workspace:…") doesn't get re-added as a
    // manual entry with id "local:…" pointing at the same physical
    // path. Whatever covers a path first wins.
    const existingRoots = new Set(
      this.descriptors.map((d) => d.root).filter((r): r is string => !!r),
    );
    const added: DatasetDescriptor[] = [];
    let skipped = 0;
    for (const p of found) {
      if (existingRoots.has(p)) {
        skipped++;
        continue;
      }
      existingRoots.add(p);
      added.push(
        this.upsert({
          id: `local:${p}`,
          name: path.basename(p),
          root: p,
          source: "manual",
        }),
      );
    }

    if (added.length > 0) {
      const note =
        skipped > 0
          ? `Added ${added.length} LeRobot dataset${added.length === 1 ? "" : "s"} (${skipped} already registered)`
          : `Added ${added.length} LeRobot dataset${added.length === 1 ? "" : "s"}`;
      void vscode.window.showInformationMessage(note);
    }
    return added;
  }

  async addSshDataset(target: SshTarget): Promise<DatasetDescriptor | undefined> {
    try {
      const descriptor = await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Fetching ${target.user ?? ""}@${target.host}:${target.remotePath}`,
        },
        async (progress) => {
          return fetchSshDataset(target, sshCacheRoot(this.context), (msg) =>
            progress.report({ message: msg }),
          );
        },
      );
      return this.upsert(descriptor);
    } catch (err) {
      logError(`adding SSH dataset ${target.host}:${target.remotePath}`, err);
      vscode.window.showErrorMessage(
        `Could not add SSH dataset: ${(err as Error).message}`,
      );
      return undefined;
    }
  }

  async addHuggingFaceRepo(repoId: string): Promise<DatasetDescriptor | undefined> {
    try {
      const descriptor = await ensureHuggingFaceDataset(repoId);
      return this.upsert(descriptor);
    } catch (err) {
      logError(`adding HF dataset ${repoId}`, err);
      vscode.window.showErrorMessage(
        `Could not add Hugging Face dataset ${repoId}: ${(err as Error).message}`,
      );
      return undefined;
    }
  }

  /**
   * Rename a local dataset folder on disk and update the descriptor.
   * Returns the updated descriptor, or undefined on failure.
   */
  async renameDataset(id: string, newName: string): Promise<DatasetDescriptor | undefined> {
    const descriptor = this.get(id);
    if (!descriptor || !descriptor.root || descriptor.source === "ssh") {
      void vscode.window.showErrorMessage("Only local datasets can be renamed.");
      return undefined;
    }
    const oldPath = descriptor.root;
    const parentDir = path.dirname(oldPath);
    const newPath = path.join(parentDir, newName);

    if (oldPath === newPath) return descriptor;
    if (await exists(newPath)) {
      void vscode.window.showErrorMessage(`A folder named "${newName}" already exists in ${parentDir}.`);
      return undefined;
    }

    try {
      await fs.rename(oldPath, newPath);
    } catch (err) {
      void vscode.window.showErrorMessage(`Failed to rename folder: ${(err as Error).message}`);
      return undefined;
    }

    // Remove old entry and add updated one (id changes with the path).
    this.descriptors = this.descriptors.filter((d) => d.id !== id);
    const updated: DatasetDescriptor = {
      ...descriptor,
      id: `local:${newPath}`,
      name: newName,
      root: newPath,
    };
    return this.upsert(updated);
  }

  remove(id: string): void {
    const before = this.descriptors.length;
    const removed = this.descriptors.find((d) => d.id === id);
    this.descriptors = this.descriptors.filter((d) => d.id !== id);
    this.snapshotCache.delete(id);
    if (this.descriptors.length !== before) {
      this.persist();
      this.refreshSshPins();
      this._onDidChange.fire();
      if (removed?.source === "ssh" && removed.ssh) {
        // The user explicitly dropped this SSH dataset — its cache
        // (downloaded meta + per-episode files) is no longer
        // referenced by anything in the tree, so reclaim the disk
        // space. Fire-and-forget; not having the cache around just
        // means a slightly slower re-add later.
        const dir = sshCacheDir(sshCacheRoot(this.context), removed.ssh);
        void fs.rm(dir, { recursive: true, force: true }).catch((err) => {
          logError(`removing SSH cache ${dir}`, err);
        });
      }
    }
  }

  /**
   * Return the adapter for a dataset id. Throws for v3.0 datasets (mutation
   * not yet supported) and when the root is not available.
   */
  private async getWritableAdapter(id: string) {
    const descriptor = this.get(id);
    if (!descriptor) throw new Error(`Unknown dataset id: ${id}`);
    if (!descriptor.root) throw new Error(`Dataset ${descriptor.name} has no local root.`);
    if (descriptor.source === "ssh") throw new Error("Task editing is not supported for SSH datasets.");
    const detection = await detectDatasetVersion(descriptor.root);
    if (detection.version === "v3.0") {
      throw new Error("Task editing is not yet supported for v3.0 datasets.");
    }
    const adapter = getAdapter(detection.version);
    if (!adapter.saveTasks || !adapter.readEpisodeRecords || !adapter.saveEpisodeRecords) {
      throw new Error("This dataset version does not support task editing.");
    }
    return { adapter, root: descriptor.root };
  }

  /** Add a new task definition. Returns the created TaskInfo. */
  async addTask(id: string, taskName: string): Promise<TaskInfo> {
    const { adapter, root } = await this.getWritableAdapter(id);
    const ctx = { root, info: await adapter.loadInfo(root) };
    const existing = await adapter.loadTasks(ctx);

    // Auto-assign next task_index.
    const nextIndex =
      existing.length > 0
        ? Math.max(...existing.map((t) => t.taskIndex)) + 1
        : 0;
    const task: TaskInfo = { taskIndex: nextIndex, task: taskName };
    existing.push(task);

    await adapter.saveTasks!(root, existing);
    this.snapshotCache.delete(id);
    log(`Added task [${task.taskIndex}] "${task.task}" to dataset ${id}`);
    return task;
  }

  /** Rename a task definition. Also updates episode references. */
  async renameTask(id: string, oldName: string, newName: string): Promise<void> {
    const { adapter, root } = await this.getWritableAdapter(id);
    const ctx = { root, info: await adapter.loadInfo(root) };
    const tasks = await adapter.loadTasks(ctx);
    const target = tasks.find((t) => t.task === oldName);
    if (!target) throw new Error(`Task "${oldName}" not found.`);
    target.task = newName;
    await adapter.saveTasks!(root, tasks);

    // Update episode references.
    const epRecords = await adapter.readEpisodeRecords!(root);
    if (epRecords) {
      let updated = 0;
      for (const rec of epRecords) {
        const episodeTasks = rec.tasks;
        if (!Array.isArray(episodeTasks)) continue;
        const idx = episodeTasks.indexOf(oldName);
        if (idx >= 0) {
          episodeTasks[idx] = newName;
          updated++;
        }
      }
      if (updated > 0) {
        await adapter.saveEpisodeRecords!(root, epRecords);
        log(`Updated "${oldName}" → "${newName}" in ${updated} episode(s)`);
      }
    }

    this.snapshotCache.delete(id);
    log(`Renamed task "${oldName}" → "${newName}" in dataset ${id}`);
  }

  /** Delete a task definition. Removes references from episodes. */
  async deleteTask(id: string, taskName: string): Promise<void> {
    const { adapter, root } = await this.getWritableAdapter(id);
    const ctx = { root, info: await adapter.loadInfo(root) };
    const tasks = await adapter.loadTasks(ctx);
    const filtered = tasks.filter((t) => t.task !== taskName);
    if (filtered.length === tasks.length) {
      throw new Error(`Task "${taskName}" not found.`);
    }
    await adapter.saveTasks!(root, filtered);

    // Remove references from episodes.
    const epRecords = await adapter.readEpisodeRecords!(root);
    if (epRecords) {
      let updated = 0;
      for (const rec of epRecords) {
        const episodeTasks = rec.tasks;
        if (!Array.isArray(episodeTasks)) continue;
        const before = episodeTasks.length;
        const newTasks = episodeTasks.filter((t: unknown) => t !== taskName);
        if (newTasks.length !== before) {
          rec.tasks = newTasks;
          updated++;
        }
      }
      if (updated > 0) {
        await adapter.saveEpisodeRecords!(root, epRecords);
        log(`Removed "${taskName}" from ${updated} episode(s)`);
      }
    }

    this.snapshotCache.delete(id);
    log(`Deleted task "${taskName}" from dataset ${id}`);
  }

  /** Change the task_index of a task. */
  async reindexTask(id: string, taskName: string, newIndex: number): Promise<void> {
    const { adapter, root } = await this.getWritableAdapter(id);
    const ctx = { root, info: await adapter.loadInfo(root) };
    const tasks = await adapter.loadTasks(ctx);
    const target = tasks.find((t) => t.task === taskName);
    if (!target) throw new Error(`Task "${taskName}" not found.`);
    if (tasks.some((t) => t.taskIndex === newIndex && t.task !== taskName)) {
      throw new Error(`Task index ${newIndex} is already in use.`);
    }
    target.taskIndex = newIndex;
    await adapter.saveTasks!(root, tasks);
    this.snapshotCache.delete(id);
    log(`Reindexed task "${taskName}" to [${newIndex}] in dataset ${id}`);
  }

  /** Set which tasks an episode belongs to. */
  async setEpisodeTasks(datasetId: string, episodeIndex: number, taskNames: string[]): Promise<void> {
    const { adapter, root } = await this.getWritableAdapter(datasetId);
    const epRecords = await adapter.readEpisodeRecords!(root);
    if (!epRecords) {
      throw new Error("No episodes metadata file found.");
    }
    const rec = epRecords.find((r) => r.episode_index === episodeIndex);
    if (!rec) {
      throw new Error(`Episode ${episodeIndex} not found in episodes metadata.`);
    }
    rec.tasks = taskNames;
    await adapter.saveEpisodeRecords!(root, epRecords);

    // Also update task_index in the parquet file so the preview's
    // TaskBand and training data see the correct task assignment.
    const snapshot = await this.getSnapshot(datasetId);
    const taskIdx = taskNames.length > 0
      ? snapshot.tasks.find((t) => t.task === taskNames[0])?.taskIndex
      : undefined;
    if (taskIdx !== undefined) {
      const info = snapshot.info;
      const ep = snapshot.episodes.find((e) => e.episodeIndex === episodeIndex);
      if (ep) {
        const chunksSize = info.chunksSize ?? 1000;
        const chunkIdx = Math.floor(episodeIndex / chunksSize);
        const dataRel = buildDataPath({
          template: info.dataPath ?? "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
          chunkIndex: chunkIdx,
          fileIndex: 0,
          episodeIndex,
        });
        const dataPath = path.join(root, dataRel);
        if (await exists(dataPath)) {
          await this.setParquetTaskIndex(dataPath, taskIdx);
        }
      }
    }

    this.snapshotCache.delete(datasetId);
    log(`Set episode ${episodeIndex} tasks to [${taskNames.join(", ")}] in dataset ${datasetId}`);
  }

  /** Rewrite the task_index column in a single-episode parquet file. */
  private async setParquetTaskIndex(parquetPath: string, taskIdx: number): Promise<void> {
    try {
      const { parquetReadObjects, asyncBufferFromFile } = await import("hyparquet");
      const pjs = require("parquetjs");
      const buffer = await asyncBufferFromFile(parquetPath);
      const rows = (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
      if (rows.length === 0) return;
      for (const r of rows) {
        r.task_index = taskIdx;
        for (const [k, v] of Object.entries(r)) {
          if (typeof v === "bigint") (r as any)[k] = Number(v);
        }
      }
      const schemaFields = buildParquetSchema(rows[0]);
      const schema = new pjs.ParquetSchema(schemaFields);
      const tmpPath = parquetPath + ".tmp";
      const writer = await pjs.ParquetWriter.openFile(schema, tmpPath, { compression: "UNCOMPRESSED" });
      for (const row of rows) await writer.appendRow(row);
      await writer.close();
      await fs.rename(tmpPath, parquetPath);
    } catch (err) {
      logError(`setParquetTaskIndex for ${parquetPath}`, err);
    }
  }

  /** Scan workspace folders for datasets up to a configurable depth. */
  async scanWorkspace(): Promise<void> {
    const config = vscode.workspace.getConfiguration("lerobotViewer");
    const maxDepth = config.get<number>("workspaceScanDepth") ?? 3;
    const folders = vscode.workspace.workspaceFolders ?? [];

    // Paths already covered by manual / HF / SSH entries; we won't
    // double-register them under a workspace: id.
    const claimedRoots = new Set(
      this.descriptors
        .filter((d) => d.source !== "workspace")
        .map((d) => d.root)
        .filter((r): r is string => !!r),
    );

    const found: DatasetDescriptor[] = [];
    for (const folder of folders) {
      await walk(folder.uri.fsPath, maxDepth, async (dir) => {
        if (await isLeRobotDataset(dir)) {
          if (!claimedRoots.has(dir)) {
            claimedRoots.add(dir);
            found.push({
              id: `workspace:${dir}`,
              name: path.basename(dir),
              root: dir,
              source: "workspace",
            });
          }
          return "skip-children";
        }
        return "continue";
      });
    }

    // Reconcile: keep manual + HF + SSH entries; replace workspace
    // entries with the freshly-discovered set so removed folders
    // disappear.
    const nonWorkspace = this.descriptors.filter((d) => d.source !== "workspace");
    this.descriptors = dedupeById([...nonWorkspace, ...found]);
    this.persist();
    this._onDidChange.fire();
    log(`Workspace scan complete: ${found.length} dataset(s) found`);
  }

  async deleteEpisode(datasetId: string, episodeIndex: number): Promise<void> {
    const descriptor = this.get(datasetId);
    if (!descriptor) throw new Error(`Unknown dataset id: ${datasetId}`);
    if (!descriptor.root) throw new Error("Dataset root is unavailable.");
    if (descriptor.source === "ssh") {
      throw new Error("Deleting episodes from SSH datasets is not supported.");
    }

    const snapshot = await this.getSnapshot(datasetId);
    const episode = snapshot.episodes.find((ep) => ep.episodeIndex === episodeIndex);
    if (!episode) throw new Error(`Episode ${episodeIndex} not found.`);

    if (snapshot.version !== "v2.0" && snapshot.version !== "v2.1") {
      throw new Error("Episode deletion is currently supported only for LeRobot v2.x datasets.");
    }

    await this.deleteV2Episode(descriptor.root, snapshot, episode);
    await this.reindexAfterDelete(descriptor.root, episodeIndex, snapshot);
    this.invalidate(datasetId);
  }

  private async reindexAfterDelete(
    root: string, deletedIdx: number, snapshot: DatasetSnapshot,
  ): Promise<void> {
    // 1. Update episodes.jsonl — remove deleted, renumber remaining.
    const epPath = path.join(root, "meta", "episodes.jsonl");
    if (!(await exists(epPath))) return;
    const epText = await fs.readFile(epPath, "utf8");
    let eps = epText.trim().split("\n")
      .map((l) => JSON.parse(l) as Record<string, unknown>)
      .filter((e) => e.episode_index !== deletedIdx);
    const oldToNew = new Map<number, number>();
    eps = eps.map((e, i) => {
      const old = e.episode_index as number;
      oldToNew.set(old, i);
      return { ...e, episode_index: i };
    });
    const { writeJsonl } = await import("./adapters/util");
    await writeJsonl(epPath, eps);

    // 2. Rename parquet + video files for episodes that got new indices.
    // The deleted episode's frame count must be subtracted from the global
    // `index` column of all subsequent episodes to keep it contiguous.
    const deletedFrameCount = snapshot.episodes.find(
      (e) => e.episodeIndex === deletedIdx,
    )?.length ?? 0;
    const chunkSize = snapshot.info.chunksSize ?? 1000;
    for (const [oldIdx, newIdx] of oldToNew) {
      if (oldIdx === newIdx) continue;
      const oldChunk = String(Math.floor(oldIdx / chunkSize)).padStart(3, "0");
      const newChunk = String(Math.floor(newIdx / chunkSize)).padStart(3, "0");
      const oldEp = String(oldIdx).padStart(6, "0");
      const newEp = String(newIdx).padStart(6, "0");
      // Rename parquet.
      const dataTpl = snapshot.info.dataPath ?? "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet";
      const oldData = path.join(root, dataTpl
        .replace("{episode_chunk:03d}", oldChunk)
        .replace("{episode_index:06d}", oldEp));
      const newData = path.join(root, dataTpl
        .replace("{episode_chunk:03d}", newChunk)
        .replace("{episode_index:06d}", newEp));
      if (await exists(oldData) && oldData !== newData) {
        await fs.mkdir(path.dirname(newData), { recursive: true });
        await fs.rename(oldData, newData);
        // Update episode_index and index columns inside the parquet.
        const indexOffset = oldIdx > deletedIdx ? -deletedFrameCount : 0;
        await this.fixEpisodeIndex(newData, newIdx, indexOffset);
      }
      // Rename videos.
      const vidTpl = snapshot.info.videoPath ?? "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4";
      for (const cam of snapshot.cameraKeys) {
        const oldVid = path.join(root, vidTpl
          .replace("{episode_chunk:03d}", oldChunk)
          .replace("{video_key}", cam)
          .replace("{episode_index:06d}", oldEp));
        const newVid = path.join(root, vidTpl
          .replace("{episode_chunk:03d}", newChunk)
          .replace("{video_key}", cam)
          .replace("{episode_index:06d}", newEp));
        if (await exists(oldVid) && oldVid !== newVid) {
          await fs.mkdir(path.dirname(newVid), { recursive: true });
          await fs.rename(oldVid, newVid);
        }
      }
    }

    // 3. Update episodes_stats.jsonl — remove deleted, renumber remaining.
    const statsPath = path.join(root, "meta", "episodes_stats.jsonl");
    if (await exists(statsPath)) {
      const statsText = await fs.readFile(statsPath, "utf8");
      let statsLines = statsText.trim().split("\n")
        .map((l) => JSON.parse(l) as Record<string, unknown>)
        .filter((s) => s.episode_index !== deletedIdx);
      statsLines = statsLines.map((s, i) => ({ ...s, episode_index: i }));
      await writeStatsJsonl(statsPath, statsLines);
    }

    // 4. Update info.json totals and splits.
    const infoPath = path.join(root, "meta", "info.json");
    if (await exists(infoPath)) {
      const raw = JSON.parse(await fs.readFile(infoPath, "utf8")) as Record<string, unknown>;
      if (typeof raw.total_episodes === "number") raw.total_episodes = eps.length;
      const totalFrames = eps.reduce((s, e) => s + (e.length as number), 0);
      raw.total_frames = totalFrames;
      if (typeof raw.total_videos === "number") raw.total_videos = eps.length * snapshot.cameraKeys.length;
      // Update splits to reflect new episode count.
      if (raw.splits && typeof raw.splits === "object") {
        const newSplits: Record<string, string> = {};
        for (const [name, range] of Object.entries(raw.splits as Record<string, unknown>)) {
          if (typeof range === "string" && /^\d+:\d+$/.test(range)) {
            newSplits[name] = `0:${eps.length}`;
          }
        }
        if (Object.keys(newSplits).length > 0) raw.splits = newSplits;
      }
      await fs.writeFile(infoPath, JSON.stringify(raw, null, 2), "utf8");
    }
  }

  private async fixEpisodeIndex(parquetPath: string, newIndex: number, indexOffset = 0): Promise<void> {
    try {
      const { parquetReadObjects, asyncBufferFromFile } = await import("hyparquet");
      const pjs = require("parquetjs");
      const buffer = await asyncBufferFromFile(parquetPath);
      const rows = (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
      if (rows.length === 0) return;

      // Update episode_index in every row. Also shift the global `index`
      // column by the offset (negative when an earlier episode was deleted).
      for (const r of rows) {
        r.episode_index = newIndex;
        if (indexOffset !== 0 && "index" in r) {
          r.index = Number(r.index) + indexOffset;
        }
      }

      // Use buildParquetSchema to preserve correct column types
      // (INTERNAL_TYPES maps timestamp→DOUBLE, episode_index→INT64, etc.).
      // A custom schema inference here would misclassify timestamp:0 as
      // INT64 and truncate all fractional values to integers.
      const schemaFields = buildParquetSchema(rows[0]);
      const schema = new pjs.ParquetSchema(schemaFields);

      const tmpPath = parquetPath + ".tmp";
      const writer = await pjs.ParquetWriter.openFile(schema, tmpPath, { compression: "UNCOMPRESSED" });
      for (const r of rows) {
        for (const [k, v] of Object.entries(r)) {
          if (typeof v === "bigint") (r as any)[k] = Number(v);
        }
        await writer.appendRow(r);
      }
      await writer.close();
      await fs.rename(tmpPath, parquetPath);
    } catch (err) {
      logError(`fixEpisodeIndex for ${parquetPath}`, err);
    }
  }

  private async persistV2EpisodeMetadata(root: string, episodeIndex: number): Promise<void> {
    const metaFile = path.join(root, "meta", "episodes.jsonl");
    if (!(await exists(metaFile))) {
      throw new Error("Could not delete episode: v2 episode metadata file meta/episodes.jsonl is missing.");
    }

    const raw = await fs.readFile(metaFile, "utf8");
    const lines = raw.split("\n").filter((line) => line.trim().length > 0);
    const output: string[] = [];
    let removed = 0;
    for (const line of lines) {
      try {
        const obj = JSON.parse(line) as Record<string, unknown>;
        const idx = Number(obj.episode_index ?? obj.episodeIndex ?? NaN);
        if (!Number.isNaN(idx) && idx === episodeIndex) {
          removed++;
          continue;
        }
      } catch {
        // Preserve malformed lines; deleting only by matching episode index.
      }
      output.push(line);
    }

    if (removed === 0) {
      throw new Error(`Could not delete episode: episode ${episodeIndex} was not found in meta/episodes.jsonl.`);
    }
    await fs.writeFile(metaFile, output.join("\n") + (output.length > 0 ? "\n" : ""), "utf8");
  }

  private async deleteV2EpisodeFiles(root: string, snapshot: DatasetSnapshot, episode: LeRobotEpisode): Promise<void> {
    const chunkSize = snapshot.info.chunksSize ?? 1000;
    const chunkIndex = Math.floor(episode.episodeIndex / chunkSize);
    const dataTemplate = snapshot.info.dataPath ?? "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet";
    const videoTemplate = snapshot.info.videoPath ?? "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4";
    const dataRel = buildDataPath({
      template: dataTemplate,
      chunkIndex,
      fileIndex: 0,
      episodeIndex: episode.episodeIndex,
    });
    const candidates = [path.join(root, dataRel)];

    for (const videoKey of snapshot.cameraKeys) {
      const videoRel = buildVideoPath({
        template: videoTemplate,
        chunkIndex,
        fileIndex: 0,
        episodeIndex: episode.episodeIndex,
        videoKey,
      });
      candidates.push(path.join(root, videoRel));
    }

    for (const candidate of candidates) {
      if (await exists(candidate)) {
        await fs.rm(candidate, { force: true });
      }
    }
  }

  private async deleteV2Episode(root: string, snapshot: DatasetSnapshot, episode: LeRobotEpisode): Promise<void> {
    await this.deleteV2EpisodeFiles(root, snapshot, episode);
    await this.persistV2EpisodeMetadata(root, episode.episodeIndex);
  }

  dispose(): void {
    this._onDidChange.dispose();
  }

  // ---------------- internal ----------------

  private upsert(descriptor: DatasetDescriptor): DatasetDescriptor {
    const idx = this.descriptors.findIndex((d) => d.id === descriptor.id);
    if (idx >= 0) {
      this.descriptors[idx] = descriptor;
    } else {
      this.descriptors = [...this.descriptors, descriptor];
    }
    this.snapshotCache.delete(descriptor.id);
    this.persist();
    this.refreshSshPins();
    this._onDidChange.fire();
    return descriptor;
  }

  /**
   * Tell the SSH connection pool which targets correspond to currently
   * registered datasets, so their sessions can stay alive past the
   * idle timeout. Called whenever the descriptor list mutates.
   */
  private refreshSshPins(): void {
    const targets = this.descriptors
      .filter((d) => d.source === "ssh" && d.ssh)
      .map((d) => d.ssh!);
    setPinnedTargets(targets);
  }

  /**
   * Sweep the SSH cache root and delete every (host, path) directory
   * that doesn't map to a currently registered SSH dataset. Run once
   * on activate; cheap because we only stat top two directory levels.
   */
  private async cleanOrphanSshCaches(): Promise<void> {
    const cacheRoot = sshCacheRoot(this.context);
    let hostDirs: string[];
    try {
      hostDirs = await fs.readdir(cacheRoot);
    } catch {
      return;
    }
    const registered = new Set(
      this.descriptors
        .filter((d) => d.source === "ssh" && d.ssh)
        .map((d) => sshCacheDir(cacheRoot, d.ssh!)),
    );
    let removed = 0;
    for (const host of hostDirs) {
      const hostPath = path.join(cacheRoot, host);
      let pathDirs: string[];
      try {
        const stat = await fs.stat(hostPath);
        if (!stat.isDirectory()) continue;
        pathDirs = await fs.readdir(hostPath);
      } catch {
        continue;
      }
      for (const p of pathDirs) {
        const full = path.join(hostPath, p);
        if (registered.has(full)) continue;
        try {
          await fs.rm(full, { recursive: true, force: true });
          removed++;
          log(`Cleaned orphan SSH cache: ${full}`);
        } catch (err) {
          logError(`cleaning orphan SSH cache ${full}`, err);
        }
      }
      // Collapse now-empty host dir.
      try {
        const remaining = await fs.readdir(hostPath);
        if (remaining.length === 0) await fs.rmdir(hostPath);
      } catch {
        // ignore
      }
    }
    if (removed > 0) log(`SSH cache: cleaned ${removed} orphan dir(s)`);
  }

  /**
   * Delete cached SSH dataset directories that haven't been touched
   * within STALE_CACHE_THRESHOLD_MS. "Touched" means a successful
   * fetchSshDataset / ensureSshFile call refreshed the cache root's
   * `.last-access` sentinel; cache dirs older than that lose their
   * meta + downloaded files (the descriptor stays in the tree, so
   * the next open just re-downloads).
   *
   * Runs in the background on activate. Legacy caches (older than
   * this feature) get a sentinel written on first sight rather than
   * being instantly deleted, so users don't lose data on the upgrade.
   */
  private async cleanStaleSshCaches(): Promise<void> {
    const cacheRoot = sshCacheRoot(this.context);
    const now = Date.now();
    let removed = 0;
    for (const d of this.descriptors) {
      if (d.source !== "ssh" || !d.ssh) continue;
      const dir = sshCacheDir(cacheRoot, d.ssh);
      const sentinel = path.join(dir, SSH_CACHE_LAST_ACCESS);
      let mtime: number;
      try {
        const stat = await fs.stat(sentinel);
        mtime = stat.mtimeMs;
      } catch {
        // No sentinel yet. Treat this as a freshly-discovered cache —
        // write the sentinel with the current time and skip deletion.
        try {
          await fs.writeFile(sentinel, "");
        } catch {
          // ignore
        }
        continue;
      }
      if (now - mtime <= STALE_CACHE_THRESHOLD_MS) continue;
      try {
        await fs.rm(dir, { recursive: true, force: true });
        this.snapshotCache.delete(d.id);
        removed++;
        log(
          `SSH cache: removed stale ${dir} (idle for ${Math.round((now - mtime) / 3_600_000)}h)`,
        );
      } catch (err) {
        logError(`cleaning stale SSH cache ${dir}`, err);
      }
    }
    if (removed > 0) log(`SSH cache: cleaned ${removed} stale dir(s)`);
  }

  /**
   * Manually clear every SSH cache directory under globalStorage.
   * Triggered by the "Clean SSH cache" command. Currently-registered
   * datasets keep their entries in the tree but lose their cached
   * meta — the next open re-downloads from the remote.
   */
  async cleanAllSshCaches(): Promise<{ removed: number; total: number }> {
    const cacheRoot = sshCacheRoot(this.context);
    let hostDirs: string[];
    try {
      hostDirs = await fs.readdir(cacheRoot);
    } catch {
      return { removed: 0, total: 0 };
    }
    let total = 0;
    let removed = 0;
    for (const host of hostDirs) {
      const hostPath = path.join(cacheRoot, host);
      let pathDirs: string[];
      try {
        const stat = await fs.stat(hostPath);
        if (!stat.isDirectory()) continue;
        pathDirs = await fs.readdir(hostPath);
      } catch {
        continue;
      }
      for (const p of pathDirs) {
        total++;
        const full = path.join(hostPath, p);
        try {
          await fs.rm(full, { recursive: true, force: true });
          removed++;
        } catch (err) {
          logError(`cleaning SSH cache ${full}`, err);
        }
      }
      try {
        const remaining = await fs.readdir(hostPath);
        if (remaining.length === 0) await fs.rmdir(hostPath);
      } catch {
        // ignore
      }
    }
    // Invalidate every cached snapshot — their roots no longer have
    // meta on disk.
    this.snapshotCache.clear();
    log(`SSH cache: cleaned ${removed}/${total} dir(s) on manual sweep`);
    return { removed, total };
  }

  private readPersisted(): DatasetDescriptor[] {
    const raw = this.context.globalState.get<DatasetDescriptor[]>(STATE_KEY) ?? [];
    return raw.filter(isPlainDescriptor);
  }

  private persist(): void {
    // Only persist manual + huggingface entries; workspace ones are
    // recomputed on each session.
    const persistable = this.descriptors.filter((d) => d.source !== "workspace");
    void this.context.globalState.update(STATE_KEY, persistable);
  }
}

function isPlainDescriptor(value: unknown): value is DatasetDescriptor {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === "string" && typeof v.name === "string" && typeof v.source === "string";
}

function dedupeById(items: DatasetDescriptor[]): DatasetDescriptor[] {
  const seen = new Set<string>();
  return items.filter((d) => {
    if (seen.has(d.id)) return false;
    seen.add(d.id);
    return true;
  });
}

// Bounded local BFS used as a fallback when the user picks a folder
// that isn't itself a dataset. Mirrors the SSH scan's behavior so the
// two flows feel the same: depth-limited, skips noise dirs, recurses
// past dataset boundaries to catch nested datasets, but prunes the
// dataset's own internal layout (data/videos/images/meta) to avoid
// burning time on chunk subdirs.

const LOCAL_SCAN_MAX_DEPTH = 4;
const LOCAL_SCAN_MAX_RESULTS = 100;
const LOCAL_SCAN_MAX_DIRS = 5000;
const DATASET_INTERNAL_DIRS = new Set(["data", "videos", "images", "meta"]);

interface LocalScanState {
  scanned: number;
  found: number;
  currentDir: string;
}

async function findLocalDatasets(
  rootDir: string,
  token: vscode.CancellationToken,
  onProgress: (s: LocalScanState) => void,
): Promise<string[]> {
  const found: string[] = [];
  let level: Array<{ dir: string; insideDataset: boolean }> = [
    { dir: rootDir, insideDataset: false },
  ];
  let depth = 0;
  let scanned = 0;

  const shouldStop = (): boolean =>
    token.isCancellationRequested ||
    found.length >= LOCAL_SCAN_MAX_RESULTS ||
    scanned >= LOCAL_SCAN_MAX_DIRS;

  while (level.length > 0 && depth <= LOCAL_SCAN_MAX_DEPTH) {
    if (shouldStop()) break;
    const nextLevel: typeof level = [];

    for (const item of level) {
      if (shouldStop()) break;
      scanned++;
      const isDs = await isLeRobotDataset(item.dir);
      if (isDs && found.length < LOCAL_SCAN_MAX_RESULTS) found.push(item.dir);
      onProgress({ scanned, found: found.length, currentDir: item.dir });

      let entries: import("node:fs").Dirent[];
      try {
        entries = await fs.readdir(item.dir, { withFileTypes: true });
      } catch {
        continue;
      }

      const childInsideDataset = isDs || item.insideDataset;
      for (const e of entries) {
        if (!e.isDirectory()) continue;
        if (e.name.startsWith(".")) continue;
        if (SKIP_DIR_NAMES.has(e.name)) continue;
        if (childInsideDataset && DATASET_INTERNAL_DIRS.has(e.name)) continue;
        nextLevel.push({
          dir: path.join(item.dir, e.name),
          insideDataset: childInsideDataset,
        });
      }
    }

    level = nextLevel;
    depth++;
  }

  return found;
}

function shortenLocalProgress(full: string, root: string): string {
  const rel = path.relative(root, full);
  const display = rel.length === 0 ? "." : rel;
  return display.length > 60 ? "…" + display.slice(-58) : display;
}

type WalkDecision = "continue" | "skip-children";

async function walk(
  root: string,
  maxDepth: number,
  visit: (dir: string, depth: number) => Promise<WalkDecision>,
): Promise<void> {
  const queue: Array<{ dir: string; depth: number }> = [{ dir: root, depth: 0 }];
  while (queue.length) {
    const { dir, depth } = queue.shift()!;
    let decision: WalkDecision;
    try {
      decision = await visit(dir, depth);
    } catch (err) {
      logError(`walk visit ${dir}`, err);
      continue;
    }
    if (decision === "skip-children") continue;
    if (depth >= maxDepth) continue;

    let entries: import("node:fs").Dirent[];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (entry.name.startsWith(".")) continue;
      if (SKIP_DIR_NAMES.has(entry.name)) continue;
      queue.push({ dir: path.join(dir, entry.name), depth: depth + 1 });
    }
  }
}
