// Dataset merge — combines multiple v2.x LeRobot datasets into a single new
// dataset directory. Files are physically copied (not linked) so the result
// is self-contained and portable.
//
// Supported: v2.0 / v2.1 → v2.1
// Not yet supported: v3.0, SSH datasets

import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { DatasetSnapshot, LeRobotEpisode, TaskInfo } from "../types";
import {
  buildDataPath,
  buildVideoPath,
  exists,
  readJsonlIfExists,
  writeJsonl,
} from "./adapters/util";
import { writeStatsJsonl } from "./statsJson";
import { buildParquetSchema } from "./parquetSchema";

// Lazy imports — hyparquet (ESM), parquetjs (CJS).
let hyparquetPromise: Promise<typeof import("hyparquet")> | undefined;
function getHyparquet() {
  return (hyparquetPromise ??= import("hyparquet"));
}
function getParquetjs(): any {
  return require("parquetjs");
}

// ---- public API ----

export interface MergeProgress {
  /** Episodes processed so far. */
  done: number;
  /** Total episodes across all sources. */
  total: number;
  /** Human-readable description of what is happening right now. */
  current: string;
}

export interface MergeResult {
  totalEpisodes: number;
  totalFrames: number;
  totalTasks: number;
}

/**
 * Merge one or more v2.x dataset snapshots into `targetRoot`. The target
 * directory is created if it doesn't exist.
 *
 * Throws when sources are incompatible (different fps, incompatible action /
 * state shapes, mixed versions).
 */
export async function mergeDatasets(
  snapshots: DatasetSnapshot[],
  targetRoot: string,
  onProgress: (p: MergeProgress) => void,
): Promise<MergeResult> {
  if (snapshots.length < 2) {
    throw new Error("At least 2 datasets are required to merge.");
  }

  // ---- validate compatibility (fps, shapes, AND parquet schema types) ----
  validateCompatibility(snapshots);
  await validateParquetSchemas(snapshots);

  const chunksSize = snapshots[0].info.chunksSize ?? 1000;
  const fps = snapshots[0].info.fps;
  const allCameraKeys = unionCameraKeys(snapshots);
  const mergedTasks = mergeTasks(snapshots);
  const mergedEpisodes = reindexEpisodes(snapshots);

  // Build task_index remapping for each source: old_index → new_index.
  // When datasets are merged, the task list is reindexed. Episodes'
  // parquet data still has the old task_index, which now points to a
  // different (or wrong) task in the merged list.
  const taskRemap = new Map<string, Map<number, number>>();
  for (const snap of snapshots) {
    const remap = new Map<number, number>();
    for (const srcTask of snap.tasks) {
      const newIdx = mergedTasks.findIndex((t) => t.task === srcTask.task);
      if (newIdx >= 0) remap.set(srcTask.taskIndex, newIdx);
    }
    taskRemap.set(snap.descriptor.id, remap);
  }
  const totalFrames = mergedEpisodes.reduce((sum, e) => sum + e.length, 0);

  // ---- prepare target ----
  await fs.mkdir(path.join(targetRoot, "meta"), { recursive: true });
  await fs.mkdir(path.join(targetRoot, "data"), { recursive: true });
  await fs.mkdir(path.join(targetRoot, "videos"), { recursive: true });

  // ---- copy parquet + video files ----
  const total = mergedEpisodes.length;
  let done = 0;

  for (const ep of mergedEpisodes) {
    const srcSnapshot = ep._srcSnapshot!;
    const srcEp = ep._srcEpisode!;

    // ---- data parquet (rewrite with correct episode_index, index, task_index) ----
    const srcDataPath = resolveSourceDataPath(srcSnapshot, srcEp);
    if (srcDataPath && (await exists(srcDataPath))) {
      const dstDataRel = buildDataPath({
        template: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        chunkIndex: Math.floor(ep.episodeIndex / chunksSize),
        fileIndex: 0,
        episodeIndex: ep.episodeIndex,
      });
      const dstDataPath = path.join(targetRoot, dstDataRel);
      onProgress({ done, total, current: `Copying parquet for episode ${ep.episodeIndex}` });
      await fs.mkdir(path.dirname(dstDataPath), { recursive: true });
      const remap = taskRemap.get(srcSnapshot.descriptor.id);
      // Offset the global `index` column by this source's cumulative frame count.
      const indexOffset = getSourceFrameOffset(snapshots, srcSnapshot);
      await copyParquetWithEpisodeIndex(srcDataPath, dstDataPath, ep.episodeIndex, remap, indexOffset);
    }

    // ---- video files ----
    for (const camKey of allCameraKeys) {
      const srcVideo = await resolveSourceVideo(srcSnapshot, srcEp, camKey);
      if (srcVideo) {
        const dstVideoRel = buildVideoPath({
          template: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
          chunkIndex: Math.floor(ep.episodeIndex / chunksSize),
          fileIndex: 0,
          episodeIndex: ep.episodeIndex,
          videoKey: camKey,
        });
        const dstVideoPath = path.join(targetRoot, dstVideoRel);
        onProgress({ done, total, current: `Copying video ${camKey} for episode ${ep.episodeIndex}` });
        await fs.mkdir(path.dirname(dstVideoPath), { recursive: true });
        await fs.cp(srcVideo, dstVideoPath);
      }
    }

    done++;
    onProgress({ done, total, current: `Episode ${ep.episodeIndex} complete` });
  }

  // ---- write info.json ----
  const firstInfo = snapshots[0].info;
  const info: Record<string, unknown> = {
    ...firstInfo.raw,
    codebase_version: firstInfo.codebaseVersion ?? "v2.1",
    splits: { train: `0:${total}` },
    fps,
    total_episodes: total,
    total_frames: totalFrames,
    total_tasks: mergedTasks.length,
    total_videos: total * allCameraKeys.length,
    chunks_size: chunksSize,
    data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    video_path: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    features: buildMergedFeatures(snapshots, allCameraKeys),
  };
  await fs.writeFile(
    path.join(targetRoot, "meta", "info.json"),
    JSON.stringify(info, null, 2),
    "utf8",
  );

  // ---- write episodes.jsonl ----
  const epRecords = mergedEpisodes.map((ep) => ({
    episode_index: ep.episodeIndex,
    tasks: ep.tasks,
    length: ep.length,
  }));
  await writeJsonl(path.join(targetRoot, "meta", "episodes.jsonl"), epRecords);

  // ---- write tasks.jsonl ----
  const taskRecords = mergedTasks.map((t) => ({
    task_index: t.taskIndex,
    task: t.task,
  }));
  await writeJsonl(path.join(targetRoot, "meta", "tasks.jsonl"), taskRecords);

  // ---- merge per-episode stats (re-index only, no recomputation) ----
  // Collect stat feature keys AND their fields per source for consistency.
  const sourceStatInfo: Array<{
    name: string;
    features: Map<string, Set<string>>; // featureKey → Set of field names
  }> = [];
  for (const snap of snapshots) {
    const srcStats = await readJsonlIfExists(
      path.join(snap.descriptor.root!, "meta", "episodes_stats.jsonl"),
    );
    const features = new Map<string, Set<string>>();
    if (srcStats && srcStats.length > 0) {
      for (const rec of srcStats) {
        const s = (rec as Record<string, unknown>).stats ?? rec;
        for (const [fk, fv] of Object.entries(s as Record<string, Record<string, unknown>>)) {
          if (!fv || typeof fv !== "object") continue;
          if (!features.has(fk)) features.set(fk, new Set());
          const fields = features.get(fk)!;
          for (const field of Object.keys(fv)) fields.add(field);
        }
      }
    }
    sourceStatInfo.push({ name: snap.descriptor.name, features });
  }
  // Check all sources have the same stat feature keys AND fields.
  if (sourceStatInfo.length >= 2) {
    const ref = sourceStatInfo[0];
    for (let i = 1; i < sourceStatInfo.length; i++) {
      const cur = sourceStatInfo[i];
      const curName = cur.name;
      // Check feature keys.
      const refKeys = new Set(ref.features.keys());
      const curKeys = new Set(cur.features.keys());
      const missingKeys = [...refKeys].filter((k) => !curKeys.has(k));
      const extraKeys = [...curKeys].filter((k) => !refKeys.has(k));
      if (missingKeys.length > 0 || extraKeys.length > 0) {
        const parts: string[] = [];
        if (missingKeys.length > 0) parts.push(`"${curName}" missing features: ${missingKeys.join(", ")}`);
        if (extraKeys.length > 0) parts.push(`"${curName}" extra features: ${extraKeys.join(", ")}`);
        throw new Error(
          `Cannot merge: stat features are inconsistent. ${parts.join("; ")}. ` +
          `Run "Recompute Stats" on the inconsistent dataset before merging.`,
        );
      }
      // Check fields within each feature.
      for (const fk of refKeys) {
        const refFields = ref.features.get(fk)!;
        const curFields = cur.features.get(fk)!;
        const missingFields = [...refFields].filter((f) => !curFields.has(f));
        const extraFields = [...curFields].filter((f) => !refFields.has(f));
        if (missingFields.length > 0 || extraFields.length > 0) {
          const parts: string[] = [];
          if (missingFields.length > 0) parts.push(`"${curName}" ${fk} missing: ${missingFields.join(", ")}`);
          if (extraFields.length > 0) parts.push(`"${curName}" ${fk} extra: ${extraFields.join(", ")}`);
          throw new Error(
            `Cannot merge: stat fields are inconsistent. ${parts.join("; ")}. ` +
            `Run "Recompute Stats" on the inconsistent dataset before merging.`,
          );
        }
      }
    }
  }

  const allEpStats: Record<string, unknown>[] = [];
  let epOffset = 0;
  for (const snap of snapshots) {
    const srcStats = await readJsonlIfExists(
      path.join(snap.descriptor.root!, "meta", "episodes_stats.jsonl"),
    );
    if (srcStats) {
      for (const rec of srcStats) {
        const statsObj = (rec as Record<string, unknown>).stats ?? rec;
        allEpStats.push({ episode_index: epOffset++, stats: statsObj });
      }
    } else {
      epOffset += snap.episodes.length;
    }
  }
  if (allEpStats.length > 0) {
    const dropped = normalizeStatsFields(allEpStats);
    // Log diagnostics.
    const sampleRec = allEpStats[0];
    const sampleKeys = sampleRec ? Object.keys((sampleRec as any).stats ?? {}).slice(0, 3) : [];
    onProgress({ done: total, total, current: `Stats merged: ${allEpStats.length} eps, ${sampleKeys.length} features${dropped.length ? `, dropped ${dropped.join(",")}` : ""}` });
    if (dropped.length > 0) {
      onProgress({ done: total, total, current: `Warn: dropped inconsistent stat fields: ${dropped.join(", ")}` });
    }
    await writeStatsJsonl(path.join(targetRoot, "meta", "episodes_stats.jsonl"), allEpStats);
  }

  return { totalEpisodes: total, totalFrames, totalTasks: mergedTasks.length };
}

// ---- validation ----

async function validateParquetSchemas(snapshots: DatasetSnapshot[]): Promise<void> {
  // Sample the first episode's parquet from each source and compare column types.
  const { parquetReadObjects, asyncBufferFromFile } = await import("hyparquet");
  const sources: Array<{ name: string; types: Record<string, string> }> = [];
  for (const snap of snapshots) {
    const root = snap.descriptor.root!;
    const firstEp = snap.episodes[0];
    const dataRel = buildDataPath({
      template: snap.info.dataPath ?? "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
      chunkIndex: Math.floor(firstEp.episodeIndex / (snap.info.chunksSize ?? 1000)),
      fileIndex: 0,
      episodeIndex: firstEp.episodeIndex,
    });
    const dataPath = path.join(root, dataRel);
    if (!(await exists(dataPath))) continue;
    try {
      const buffer = await asyncBufferFromFile(dataPath);
      const rows = (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
      if (rows.length === 0) continue;
      const row = rows[0];
      const types: Record<string, string> = {};
      for (const [k, v] of Object.entries(row)) {
        if (v === null || v === undefined) continue;
        types[k] = Array.isArray(v)
          ? `list<${typeof (v as unknown[])[0]}>`
          : typeof v === "bigint" ? "int64" : typeof v;
      }
      sources.push({ name: snap.descriptor.name, types });
    } catch { /* skip */ }
  }
  if (sources.length < 2) return;
  // Compare each source against the first.
  const base = sources[0];
  for (let i = 1; i < sources.length; i++) {
    const s = sources[i];
    for (const [col, baseType] of Object.entries(base.types)) {
      const sType = s.types[col];
      if (sType && !typesCompatible(baseType, sType)) {
        throw new Error(
          `Parquet type mismatch for column "${col}": ` +
          `"${base.name}" has ${baseType}, "${s.name}" has ${sType}. ` +
          `Use "Drop Dimensions" or "Delete Feature" on the inconsistent dataset to normalize types before merging.`,
        );
      }
    }
  }
}

/**
 * Two parquet column types are compatible when they represent the same
 * logical data.  `number` (hyparquet reads DOUBLE columns as JS number)
 * and `int64` (hyparquet reads INT64 columns as BigInt) are both numeric
 * — the physical storage differs but the values are interoperable.
 * List element types are also compared loosely for the same reason.
 */
function typesCompatible(a: string, b: string): boolean {
  if (a === b) return true;
  // Normalize numeric representations.
  const numeric = new Set(["number", "int64"]);
  if (numeric.has(a) && numeric.has(b)) return true;
  // Normalize list element types (e.g. list<number> vs list<int64>).
  const listMatchA = a.match(/^list<(.+)>$/);
  const listMatchB = b.match(/^list<(.+)>$/);
  if (listMatchA && listMatchB) {
    return typesCompatible(listMatchA[1], listMatchB[1]);
  }
  return false;
}

function validateCompatibility(snapshots: DatasetSnapshot[]): void {
  const base = snapshots[0];
  if (base.version !== "v2.0" && base.version !== "v2.1") {
    throw new Error("Only v2.0 / v2.1 datasets can be merged.");
  }

  const baseStateShape = JSON.stringify(base.info.features["observation.state"]?.shape);
  const baseActionShape = JSON.stringify(base.info.features["action"]?.shape);

  for (let i = 1; i < snapshots.length; i++) {
    const s = snapshots[i];
    if (s.version !== "v2.0" && s.version !== "v2.1") {
      throw new Error(
        `Dataset "${s.descriptor.name}" is ${s.version}; only v2.x datasets can be merged.`,
      );
    }
    if (s.info.fps !== base.info.fps) {
      throw new Error(
        `FPS mismatch: "${base.descriptor.name}" is ${base.info.fps}fps but ` +
          `"${s.descriptor.name}" is ${s.info.fps}fps.`,
      );
    }
    const sStateShape = JSON.stringify(s.info.features["observation.state"]?.shape);
    if (sStateShape !== baseStateShape) {
      throw new Error(
        `State shape mismatch between "${base.descriptor.name}" and "${s.descriptor.name}".`,
      );
    }
    const sActionShape = JSON.stringify(s.info.features["action"]?.shape);
    if (sActionShape !== baseActionShape) {
      throw new Error(
        `Action shape mismatch between "${base.descriptor.name}" and "${s.descriptor.name}".`,
      );
    }
  }
}

// ---- episode re-indexing ----

interface MergeEpisode extends LeRobotEpisode {
  _srcSnapshot: DatasetSnapshot;
  _srcEpisode: LeRobotEpisode;
}

function reindexEpisodes(snapshots: DatasetSnapshot[]): MergeEpisode[] {
  const out: MergeEpisode[] = [];
  let nextIndex = 0;
  for (const snap of snapshots) {
    const sorted = [...snap.episodes].sort((a, b) => a.episodeIndex - b.episodeIndex);
    for (const ep of sorted) {
      out.push({
        episodeIndex: nextIndex++,
        tasks: ep.tasks,
        length: ep.length,
        _srcSnapshot: snap,
        _srcEpisode: ep,
      });
    }
  }
  return out;
}

/**
 * Remove stat fields (like q01/q99) that aren't present in ALL records.
 * Returns the list of dropped field names (feature.field).
 */
function normalizeStatsFields(records: Record<string, unknown>[]): string[] {
  const dropped: string[] = [];
  // For each feature, find fields that exist in some but not all records.
  const featureFields = new Map<string, Map<string, number>>();
  for (const rec of records) {
    const stats = (rec.stats ?? rec) as Record<string, Record<string, unknown>>;
    for (const [fk, fv] of Object.entries(stats)) {
      if (!fv || typeof fv !== "object") continue;
      if (!featureFields.has(fk)) featureFields.set(fk, new Map());
      const fieldCount = featureFields.get(fk)!;
      for (const k of Object.keys(fv)) {
        fieldCount.set(k, (fieldCount.get(k) ?? 0) + 1);
      }
    }
  }
  const totalRecords = records.length;
  for (const [fk, fieldCount] of featureFields) {
    for (const [field, count] of fieldCount) {
      if (count < totalRecords) {
        dropped.push(`${fk}.${field}`);
      }
    }
  }
  // Remove inconsistent fields per-feature from all records.
  if (dropped.length > 0) {
    // Group by feature: featureKey → Set of field names to drop.
    const byFeature = new Map<string, Set<string>>();
    for (const d of dropped) {
      // Feature keys may contain dots (e.g. "observation.images.cam_high").
      // Split on the LAST dot to separate feature key from field name.
      const lastDot = d.lastIndexOf(".");
      const fk = d.slice(0, lastDot);
      const field = d.slice(lastDot + 1);
      if (!byFeature.has(fk)) byFeature.set(fk, new Set());
      byFeature.get(fk)!.add(field);
    }
    for (const rec of records) {
      const stats = (rec.stats ?? rec) as Record<string, Record<string, unknown>>;
      for (const [fk, fields] of byFeature) {
        const feat = stats[fk] as Record<string, unknown> | undefined;
        if (!feat) continue;
        for (const f of fields) delete feat[f];
      }
    }
  }
  // Also normalize types: ensure count is int64, numeric arrays are consistent.
  for (const rec of records) {
    const stats = (rec.stats ?? rec) as Record<string, Record<string, unknown>>;
    for (const feat of Object.values(stats)) {
      if (!feat || typeof feat !== "object") continue;
      const f = feat as Record<string, unknown>;
      // count should always be an integer array.
      if ("count" in f && f.count !== null) {
        const arr = Array.isArray(f.count) ? f.count : [f.count];
        f.count = mapLeaves(arr, (v: unknown) => Math.round(Number(v)));
      }
      // Ensure all numeric arrays are plain numbers (no BigInt, no mixed types).
      // Use recursive mapLeaves to preserve video stats' 3-level [[[r]],[[g]],[[b]]] shape.
      for (const k of ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]) {
        if (k in f && f[k] !== null && Array.isArray(f[k])) {
          f[k] = mapLeaves(f[k] as unknown[], (v: unknown) => Number(v));
        }
      }
    }
  }

  return dropped;
}

/** Recursively apply `fn` to every leaf (non-array) value in a nested array. */
function mapLeaves(arr: unknown[], fn: (v: unknown) => unknown): unknown[] {
  return arr.map((v) => (Array.isArray(v) ? mapLeaves(v as unknown[], fn) : fn(v)));
}

// ---- task merging ----

function mergeTasks(snapshots: DatasetSnapshot[]): TaskInfo[] {
  const seen = new Map<string, number>(); // task name → assigned task_index
  let nextIdx = 0;
  for (const snap of snapshots) {
    for (const t of snap.tasks) {
      if (!seen.has(t.task)) {
        seen.set(t.task, nextIdx++);
      }
    }
  }
  return [...seen.entries()].map(([task, taskIndex]) => ({ taskIndex, task }));
}

// ---- camera keys ----

function unionCameraKeys(snapshots: DatasetSnapshot[]): string[] {
  const keys = new Set<string>();
  for (const s of snapshots) {
    for (const k of s.cameraKeys) keys.add(k);
  }
  return [...keys].sort();
}

/**
 * Copy a v2.x per-episode parquet file, rewriting every row's `episode_index`
 * to `newIndex`.  Without this, merged datasets reuse the old internal
 * episode_index values, causing duplicates (two files with ep_index=0) and
 * missing episodes (no file with ep_index=60).
 */
/**
 * Compute the total frame count of all snapshots before `target`.
 * Used to offset the global `index` column during merge.
 */
function getSourceFrameOffset(
  snapshots: DatasetSnapshot[],
  target: DatasetSnapshot,
): number {
  let offset = 0;
  for (const snap of snapshots) {
    if (snap.descriptor.id === target.descriptor.id) break;
    offset += snap.episodes.reduce((s, e) => s + (e.length || 0), 0);
  }
  return offset;
}

async function copyParquetWithEpisodeIndex(
  srcPath: string,
  dstPath: string,
  newIndex: number,
  taskRemap?: Map<number, number>,
  indexOffset = 0,
): Promise<void> {
  try {
    const { parquetReadObjects, asyncBufferFromFile } = await getHyparquet();
    const pjs = getParquetjs();

    const buffer = await asyncBufferFromFile(srcPath);
    const rows = (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];

    // Update episode_index, index, task_index, and convert BigInt to Number.
    for (const r of rows) {
      r.episode_index = newIndex;
      for (const [k, v] of Object.entries(r)) {
        if (typeof v === "bigint") (r as any)[k] = Number(v);
      }
      if (taskRemap && "task_index" in r) {
        const oldTi = Number(r.task_index);
        if (taskRemap.has(oldTi)) r.task_index = taskRemap.get(oldTi)!;
      }
      if ("index" in r && indexOffset > 0) {
        r.index = Number(r.index) + indexOffset;
      }
    }

    // Preserve original schema types via buildParquetSchema.
    const schemaFields = buildParquetSchema(rows[0]);
    const schema = new pjs.ParquetSchema(schemaFields);
    const writer = await pjs.ParquetWriter.openFile(schema, dstPath, { compression: "UNCOMPRESSED" });
    for (const row of rows) await writer.appendRow(row);
    await writer.close();
  } catch {
    // Fallback: plain copy if the source isn't a valid parquet file.
    await fs.cp(srcPath, dstPath);
  }
}

// ---- feature merging ----

function buildMergedFeatures(
  snapshots: DatasetSnapshot[],
  _cameraKeys: string[],
): Record<string, unknown> {
  // Take features from the first snapshot and augment with any extra cameras.
  const features = { ...snapshots[0].info.features } as Record<string, unknown>;
  for (const snap of snapshots) {
    for (const camKey of snap.cameraKeys) {
      if (!(camKey in features)) {
        features[camKey] = snap.info.features[camKey];
      }
    }
  }
  return features;
}

// ---- path resolution for source files ----

function resolveSourceDataPath(
  snapshot: DatasetSnapshot,
  episode: LeRobotEpisode,
): string | undefined {
  const root = snapshot.descriptor.root;
  if (!root) return undefined;
  const chunksSize = snapshot.info.chunksSize ?? 1000;
  const rel = buildDataPath({
    template: snapshot.info.dataPath ?? "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    chunkIndex: Math.floor(episode.episodeIndex / chunksSize),
    fileIndex: 0,
    episodeIndex: episode.episodeIndex,
  });
  return path.join(root, rel);
}

async function resolveSourceVideo(
  snapshot: DatasetSnapshot,
  episode: LeRobotEpisode,
  videoKey: string,
): Promise<string | undefined> {
  const root = snapshot.descriptor.root;
  if (!root) return undefined;
  const chunksSize = snapshot.info.chunksSize ?? 1000;
  const rel = buildVideoPath({
    template:
      snapshot.info.videoPath ??
      "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    chunkIndex: Math.floor(episode.episodeIndex / chunksSize),
    fileIndex: 0,
    episodeIndex: episode.episodeIndex,
    videoKey,
  });
  const abs = path.join(root, rel);
  return (await exists(abs)) ? abs : undefined;
}
