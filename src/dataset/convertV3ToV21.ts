// Convert a LeRobot v3.0 (sharded) dataset to v2.1 (per-episode) format.
//
// This makes v3.0 datasets compatible with the extension's editing features
// (task editing, episode deletion, merge, etc.).
//
// Dependencies:
//   - hyparquet (already in project) — read v3.0 sharded parquet
//   - parquetjs             — write per-episode parquet files
//   - ffmpeg (external)      — slice shared video by timestamp

import * as cp from "node:child_process";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { LeRobotEpisode, LeRobotInfo } from "../types";
import { V30Adapter } from "./adapters/V30Adapter";
import {
  buildDataPath,
  buildVideoPath,
  exists,
  writeJsonl,
} from "./adapters/util";
import { writeStatsJsonl } from "./statsJson";

// Lazy ESM/CJS imports.
let hyparquetPromise: Promise<typeof import("hyparquet")> | undefined;
function getHyparquet() {
  return (hyparquetPromise ??= import("hyparquet"));
}

// parquetjs is a CJS module. Use a plain require (available in Node CJS
// and tsx). TypeScript sees the `require` global from @types/node.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getParquetjs(): any {
  return require("parquetjs");
}

// ---- progress / result types ----

export interface ConvertProgress {
  done: number;
  total: number;
  current: string;
}

export interface ConvertResult {
  totalEpisodes: number;
  totalFrames: number;
  warnings: string[];
}

// ---- public API ----

/**
 * Convert a v3.0 LeRobot dataset at `sourceRoot` into a v2.1 dataset at
 * `targetRoot`. The target directory is created if needed.
 *
 * @param sourceRoot  Absolute path to the v3.0 dataset root.
 * @param targetRoot  Absolute path where the v2.1 dataset will be written.
 * @param onProgress  Called after each episode finishes.
 */
export async function convertV3ToV21(
  sourceRoot: string,
  targetRoot: string,
  onProgress: (p: ConvertProgress) => void,
): Promise<ConvertResult> {
  const warnings: string[] = [];

  // 1. Load v3.0 source metadata.
  const adapter = new V30Adapter();
  const info = await adapter.loadInfo(sourceRoot);
  const episodes = await adapter.loadEpisodes({ root: sourceRoot, info });
  const tasks = await adapter.loadTasks({ root: sourceRoot, info });

  if (episodes.length === 0) {
    throw new Error("Source dataset has no episodes. Nothing to convert.");
  }

  // 2. Check ffmpeg.
  let ffmpegAvailable = false;
  try {
    await execFile("ffmpeg", ["-version"]);
    ffmpegAvailable = true;
  } catch {
    warnings.push("ffmpeg not found — video conversion will be skipped. Install ffmpeg to include videos.");
  }

  // 3. Create target directory layout.
  await fs.mkdir(path.join(targetRoot, "meta"), { recursive: true });
  await fs.mkdir(path.join(targetRoot, "data"), { recursive: true });
  await fs.mkdir(path.join(targetRoot, "videos"), { recursive: true });

  const chunksSize = info.chunksSize ?? 1000;
  const total = episodes.length;
  let totalFrames = 0;

  // Pre-built canonical schema (populated from first episode's data).
  // Using the same schema for all episodes prevents Arrow type conflicts.
  let canonicalSchema: unknown = null;

  // Cache: data shard path → loaded rows (to avoid re-reading the same shard).
  const shardCache = new Map<string, Record<string, unknown>[]>();

  // Per-episode stats records (one per line in episodes_stats.jsonl).
  const epStatsRecords: Record<string, unknown>[] = [];

  // 4. Convert each episode.
  for (let i = 0; i < episodes.length; i++) {
    const ep = episodes[i];
    const newIdx = i; // Re-index sequentially 0, 1, 2, ...
    const chunkIdx = Math.floor(newIdx / chunksSize);

    // --- data parquet ---
    const shardKey = ep.dataShard
      ? `${ep.dataShard.chunkIndex}/${ep.dataShard.fileIndex}`
      : `fallback/${ep.episodeIndex}`;

    let rows: Record<string, unknown>[] | undefined = shardCache.get(shardKey);
    if (!rows) {
      rows = await readDataShardRows(sourceRoot, info, ep, warnings);
      if (rows) shardCache.set(shardKey, rows);
    }

    if (rows) {
      // frameRange may be global, not per-file. Try slice first,
      // fall back to episode_index filter if OOB.
      let epRows = ep.frameRange
        ? rows.slice(ep.frameRange[0], ep.frameRange[1])
        : rows.filter((r) => Number(r.episode_index) === ep.episodeIndex);
      if (epRows.length === 0) {
        epRows = rows.filter((r) => Number(r.episode_index) === ep.episodeIndex);
      }
      if (epRows.length > 0) {
        totalFrames += epRows.length;
        const cleanRows = sanitizeRows(epRows);

        // Per-episode stats.
        const epStats = new StatsAccumulator();
        epStats.ingest(cleanRows);
        epStatsRecords.push({
          episode_index: newIdx,
          stats: epStats.toPerEpisode(featureKeyMap(info)),
        });

        const dstRel = buildDataPath({
          template: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
          chunkIndex: chunkIdx,
          fileIndex: 0,
          episodeIndex: newIdx,
        });
        const dstPath = path.join(targetRoot, dstRel);
        await fs.mkdir(path.dirname(dstPath), { recursive: true });
        if (!canonicalSchema) {
          canonicalSchema = buildCanonicalSchema(info, sanitizeRow(epRows[0]));
        }
        await writeParquetFile(dstPath, epRows, canonicalSchema);
      }
    }

    // --- video files ---
    if (ffmpegAvailable && ep.videoShards && ep.videoRanges) {
      const allCameras = Object.keys(ep.videoShards);
      for (const camKey of allCameras) {
        const srcRel = buildVideoPath({
          template: info.videoPath ?? "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
          chunkIndex: ep.videoShards[camKey]?.chunkIndex ?? 0,
          fileIndex: ep.videoShards[camKey]?.fileIndex ?? 0,
          episodeIndex: ep.episodeIndex,
          videoKey: camKey,
        });
        const srcPath = path.join(sourceRoot, srcRel);
        if (!(await exists(srcPath))) continue;

        const dstRel = buildVideoPath({
          template: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
          chunkIndex: chunkIdx,
          fileIndex: 0,
          episodeIndex: newIdx,
          videoKey: camKey,
        });
        const dstPath = path.join(targetRoot, dstRel);
        await fs.mkdir(path.dirname(dstPath), { recursive: true });

        const range = ep.videoRanges[camKey];
        if (range) {
          const minDur = (ep.length ?? 0) / (info.fps || 30);
          const dur = Math.max(range[1] - range[0], minDur);
          // Pass codec name from info.json instead of probing each file.
          const vcodec = (info.features[camKey]?.info as Record<string, unknown> | undefined)?.["video.codec"] as string | undefined;
          await sliceVideo(srcPath, dstPath, range[0], range[0] + dur, vcodec);
        }
      }
    } else if (ep.videoShards && Object.keys(ep.videoShards).length > 0 && !ffmpegAvailable) {
      // Only warn once.
      if (i === 0) {
        onProgress({ done: i, total, current: "Video conversion skipped (ffmpeg not available)" });
      }
    }

    onProgress({ done: i + 1, total, current: `Episode ${newIdx} (was ${ep.episodeIndex})` });
  }

  // 5. Write v2.1 metadata.
  const cameraCount = Object.values(info.features).filter((f) => f.dtype === "video").length;
  const v21Info = buildV21Info(info, episodes.length, totalFrames, tasks.length, cameraCount, chunksSize);
  await fs.writeFile(
    path.join(targetRoot, "meta", "info.json"),
    JSON.stringify(v21Info, null, 2),
    "utf8",
  );

  const epRecords = episodes.map((ep, i) => ({
    episode_index: i,
    tasks: ep.tasks,
    length: ep.length,
  }));
  await writeJsonl(path.join(targetRoot, "meta", "episodes.jsonl"), epRecords);

  const taskRecords = tasks.map((t) => ({
    task_index: t.taskIndex,
    task: t.task,
  }));
  await writeJsonl(path.join(targetRoot, "meta", "tasks.jsonl"), taskRecords);

  // Copy video feature stats from v3.0 global stats.json into each episode's stats.
  const v30StatsPath = path.join(sourceRoot, "meta", "stats.json");
  if (await exists(v30StatsPath)) {
    try {
      const v30Stats = JSON.parse(await fs.readFile(v30StatsPath, "utf8")) as Record<string, unknown>;
      const videoKeys = Object.keys(info.features).filter(
        (k) => info.features[k]?.dtype === "video",
      );
      for (const vk of videoKeys) {
        const featStats = v30Stats[vk] as Record<string, unknown> | undefined;
        if (featStats) {
          for (const rec of epStatsRecords) {
            const statsObj = (rec as Record<string, unknown>).stats as Record<string, unknown>;
            if (statsObj) statsObj[vk] = featStats;
          }
        }
      }
    } catch { /* stats.json may not exist or be unreadable */ }
  }

  // Write per-episode stats (v2.1 canonical format).
  await writeStatsJsonl(
    path.join(targetRoot, "meta", "episodes_stats.jsonl"),
    epStatsRecords,
  );

  return { totalEpisodes: episodes.length, totalFrames, warnings };
}

// ---- data parquet helpers ----

async function readDataShardRows(
  root: string,
  info: LeRobotInfo,
  episode: LeRobotEpisode,
  warnings: string[],
): Promise<Record<string, unknown>[] | undefined> {
  const chunksSize = info.chunksSize ?? 1000;
  let chunkIndex: number;
  let fileIndex: number;

  if (episode.dataShard) {
    chunkIndex = episode.dataShard.chunkIndex;
    fileIndex = episode.dataShard.fileIndex;
  } else {
    chunkIndex = Math.floor(episode.episodeIndex / chunksSize);
    fileIndex = 0;
    warnings.push(
      `Episode ${episode.episodeIndex}: no data shard info; guessing chunk ${chunkIndex} file ${fileIndex}.`,
    );
  }

  const rel = buildDataPath({
    template: info.dataPath ?? "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    chunkIndex,
    fileIndex,
    episodeIndex: episode.episodeIndex,
  });
  const abs = path.join(root, rel);

  if (!(await exists(abs))) {
    warnings.push(`Episode ${episode.episodeIndex}: data parquet not found at ${rel}.`);
    return undefined;
  }

  try {
    const { parquetReadObjects, asyncBufferFromFile } = await getHyparquet();
    const buffer = await asyncBufferFromFile(abs);
    return (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
  } catch (err) {
    warnings.push(
      `Episode ${episode.episodeIndex}: failed to read parquet ${rel}: ${(err as Error).message}`,
    );
    return undefined;
  }
}

async function writeParquetFile(
  filePath: string,
  rows: Record<string, unknown>[],
  pjsSchema: unknown,
): Promise<void> {
  if (rows.length === 0) return;
  const pjs = getParquetjs();
  const writer = await pjs.ParquetWriter.openFile(pjsSchema, filePath, {
    compression: "UNCOMPRESSED",
  });
  for (const row of rows) {
    await writer.appendRow(sanitizeRow(row));
  }
  await writer.close();
}

function buildCanonicalSchema(info: LeRobotInfo, sampleRow: Record<string, unknown>): unknown {
  const pjs = getParquetjs();
  const fields: Record<string, unknown> = {};

  // Fixed-type internal columns.
  const INTERNAL: Record<string, string> = {
    episode_index: "INT64", frame_index: "INT64", timestamp: "DOUBLE",
    index: "INT64", task_index: "INT64",
  };

  // Feature columns from info.json (derive type and repeated from features).
  for (const [key, feat] of Object.entries(info.features)) {
    if (feat.dtype === "video") continue;
    const pt = dtypeToParquetType(feat.dtype);
    const isArray = feat.shape && feat.shape.length >= 1 && feat.shape[0] > 1;
    fields[key] = { type: pt, repeated: isArray };
  }

  // Also include any extra columns from the actual data (e.g. from older converters).
  for (const key of Object.keys(sampleRow)) {
    if (fields[key] || INTERNAL[key]) continue;
    const v = sampleRow[key];
    if (v === null || v === undefined) continue;
    if (Array.isArray(v)) {
      fields[key] = { type: "DOUBLE", repeated: true };
    } else if (typeof v === "number") {
      fields[key] = { type: "DOUBLE" };
    } else if (typeof v === "boolean") {
      fields[key] = { type: "BOOLEAN" };
    } else if (typeof v === "string") {
      fields[key] = { type: "UTF8" };
    }
  }

  // Internal columns always present.
  for (const [k, t] of Object.entries(INTERNAL)) {
    fields[k] = { type: t };
  }

  return new pjs.ParquetSchema(fields);
}

/** Convert bigint values to number — parquetjs rejects bigint. */
function sanitizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (typeof value === "bigint") {
      out[key] = Number(value);
    } else if (Array.isArray(value)) {
      out[key] = value.map((v) => (typeof v === "bigint" ? Number(v) : v));
    } else {
      out[key] = value;
    }
  }
  return out;
}

function dtypeToParquetType(dtype: string): string {
  switch (dtype) {
    case "int32": return "INT32";
    case "int64": return "INT64";
    case "float32": return "FLOAT";
    case "float64":
    case "double": return "DOUBLE";
    case "bool":
    case "boolean": return "BOOLEAN";
    case "string": return "UTF8";
    default: return "DOUBLE";
  }
}

function inferElemType(feat: { dtype: string } | undefined): string {
  if (feat) return dtypeToParquetType(feat.dtype);
  return "DOUBLE";
}

// ---- video helpers ----

function execFile(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = cp.spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout?.on("data", (d: Buffer) => (stdout += d.toString()));
    proc.stderr?.on("data", (d: Buffer) => (stderr += d.toString()));
    proc.on("close", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`${cmd} exited ${code}:\n${stderr.slice(-500)}`));
    });
    proc.on("error", (err) => reject(err));
  });
}

async function sliceVideo(
  srcPath: string,
  dstPath: string,
  fromTs: number,
  toTs: number,
  codec?: string,
): Promise<void> {
  // ffmpeg 4.x can't mux AV1 into MP4 with -c copy. Use codec from
  // info.json (via parameter) to avoid a per-file ffprobe call.
  const canCopy = !codec || codec === "h264" || codec === "hevc" || codec === "mpeg4";
  // -ss after -i: -to is absolute position in the source, NOT duration.
  // Step 1: slice with -c copy (or re-encode for AV1 etc.)
  const tmpPath = dstPath + ".tmp.mp4";
  if (canCopy) {
    await execFile("ffmpeg", [
      "-i", srcPath,
      "-ss", fromTs.toFixed(6),
      "-to", toTs.toFixed(6),
      "-c", "copy",
      "-avoid_negative_ts", "make_zero",
      "-y",
      tmpPath,
    ]);
  } else {
    try {
      await execFile("ffmpeg", [
        "-i", srcPath,
        "-ss", fromTs.toFixed(6),
        "-to", toTs.toFixed(6),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        tmpPath,
      ]);
    } catch {
      await execFile("ffmpeg", [
        "-i", srcPath,
        "-ss", fromTs.toFixed(6),
        "-to", toTs.toFixed(6),
        "-c:v", "mpeg4",
        "-q:v", "2",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        tmpPath,
      ]);
    }
  }
  // Step 2: regenerate clean PTS from 0 so the file can be safely
  // concatenated later (avoids "non monotonically increasing dts").
  await execFile("ffmpeg", [
    "-i", tmpPath,
    "-c", "copy",
    "-fflags", "+genpts",
    "-reset_timestamps", "1",
    "-y",
    dstPath,
  ]);
  await fs.unlink(tmpPath).catch(() => {});
}

// ---- info.json generation ----

function buildV21Info(
  info: LeRobotInfo,
  totalEpisodes: number,
  totalFrames: number,
  totalTasks: number,
  cameraCount: number,
  chunksSize: number,
): Record<string, unknown> {
  // Convert v3.0 features to v2.1-compatible format.
  const features: Record<string, unknown> = {};
  for (const [key, feat] of Object.entries(info.features)) {
    const entry: Record<string, unknown> = { dtype: feat.dtype };
    if (feat.shape) entry.shape = feat.shape;
    if (feat.names) entry.names = feat.names;
    // Preserve video metadata (codec, fps, etc.) from v3.0.
    if (feat.dtype === "video" && feat.info) entry.info = feat.info;
    features[key] = entry;
  }

  return {
    codebase_version: "v2.1",
    robot_type: info.robotType ?? "unknown",
    fps: info.fps,
    total_episodes: totalEpisodes,
    total_frames: totalFrames,
    total_tasks: totalTasks,
    total_videos: totalEpisodes * cameraCount,
    total_chunks: Math.ceil(totalEpisodes / chunksSize),
    chunks_size: chunksSize,
    data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    video_path: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    features,
  };
}

// ---- online stats computation (min / max / mean / std) ----

function sanitizeRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  return rows.map(sanitizeRow);
}

class StatsAccumulator {
  // key → array length (=1 for scalars)
  private lengths = new Map<string, number>();
  // key → element-wise min
  private mins = new Map<string, number[]>();
  // key → element-wise max
  private maxs = new Map<string, number[]>();
  // key → element-wise mean (tracked online, Welford)
  private means = new Map<string, number[]>();
  // key → element-wise M2 (sum of squared diffs, Welford)
  private m2s = new Map<string, number[]>();
  // key → count of rows seen
  private counts = new Map<string, number>();
  // key → per-dimension value arrays for quantile computation.
  private values = new Map<string, number[][]>();

  /** Feed a batch of sanitized rows into the accumulator. */
  ingest(rows: Record<string, unknown>[]): void {
    for (const row of rows) {
      for (const [key, value] of Object.entries(row)) {
        if (value === null || value === undefined) continue;
        const arr = Array.isArray(value) ? (value as number[]) : [value as number];
        if (arr.some((v) => typeof v !== "number" || !Number.isFinite(v))) continue;

        const n = arr.length;
        if (!this.lengths.has(key)) {
          this.lengths.set(key, n);
          this.mins.set(key, [...arr]);
          this.maxs.set(key, [...arr]);
          this.means.set(key, new Array(n).fill(0));
          this.m2s.set(key, new Array(n).fill(0));
          this.counts.set(key, 0);
          this.values.set(key, Array.from({ length: n }, () => []));
        }

        const count = this.counts.get(key)! + 1;
        this.counts.set(key, count);

        const mins = this.mins.get(key)!;
        const maxs = this.maxs.get(key)!;
        const means = this.means.get(key)!;
        const m2s = this.m2s.get(key)!;
        const vals = this.values.get(key)!;

        for (let i = 0; i < n; i++) {
          const x = arr[i];
          vals[i].push(x);
          if (x < mins[i]) mins[i] = x;
          if (x > maxs[i]) maxs[i] = x;
          const delta = x - means[i];
          means[i] += delta / count;
          const delta2 = x - means[i];
          m2s[i] += delta * delta2;
        }
      }
    }
  }

  /**
   * Produce per-episode stats (no episode_index wrapper — caller adds it).
   */
  toPerEpisode(resolveKey: (pk: string) => string): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const pk of this.counts.keys()) {
      const outKey = resolveKey(pk);
      const count = this.counts.get(pk)!;
      const mean = this.means.get(pk)!;
      const std = this.m2s.get(pk)!.map((m2) => Math.sqrt(m2 / count));
      const vals = this.values.get(pk)!;
      // Pre-sort each dimension once, compute all quantiles from sorted arrays.
      const sorted = vals.map((v) => v.slice().sort((a, b) => a - b));
      const q01 = sorted.map((s) => quantileFromSorted(s, 0.01));
      const q10 = sorted.map((s) => quantileFromSorted(s, 0.10));
      const q50 = sorted.map((s) => quantileFromSorted(s, 0.50));
      const q90 = sorted.map((s) => quantileFromSorted(s, 0.90));
      const q99 = sorted.map((s) => quantileFromSorted(s, 0.99));
      out[outKey] = {
        min: this.mins.get(pk),
        max: this.maxs.get(pk),
        mean: [...mean],
        std,
        q01, q10, q50, q90, q99,
        count: [count],
      };
    }
    return out;
  }

}

function quantileFromSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

/**
 * Build a map from parquet column name → canonical feature key (from info.json).
 * Case-insensitive matching for common discrepancies like "actions" vs "action".
 */
function featureKeyMap(info: LeRobotInfo): (pk: string) => string {
  const featureKeys = Object.keys(info.features);
  return (pk: string): string => {
    if (featureKeys.includes(pk)) return pk;
    const lower = pk.toLowerCase();
    const match = featureKeys.find((fk) => fk.toLowerCase() === lower);
    return match ?? pk;
  };
}
