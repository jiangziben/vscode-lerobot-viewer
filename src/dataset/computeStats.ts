// Compute per-feature min/max/mean/std by scanning all episode parquet files.
// Writes meta/stats.json (global) and meta/episodes_stats.jsonl (per-episode).

import * as fs from "node:fs/promises";
import * as path from "node:path";
import { V21Adapter } from "./adapters/V21Adapter";
import { V30Adapter } from "./adapters/V30Adapter";
import { exists, readJson, buildDataPath } from "./adapters/util";
import { computeVideoFeatureStats, ffmpegAvailable } from "./videoStats";
import { writeStatsJsonl, floatifyArraysInJson } from "./statsJson";
import type { LeRobotInfo, LeRobotEpisode } from "../types";

let hyparquetPromise: Promise<typeof import("hyparquet")> | undefined;
function getHyparquet() {
  return (hyparquetPromise ??= import("hyparquet"));
}

export interface StatsProgress {
  done: number;
  total: number;
}

export async function recomputeStats(
  root: string,
  onProgress: (p: StatsProgress) => void,
): Promise<string[]> {
  const warnings: string[] = [];
  // Detect version and use appropriate adapter.
  const { detectDatasetVersion } = await import("./DatasetVersionDetector");
  const version = (await detectDatasetVersion(root)).version;
  const isV3 = version === "v3.0";

  let info: LeRobotInfo;
  let episodes: LeRobotEpisode[];
  let resolveKey: (pk: string) => string;
  let readEpisodeRows: (ep: LeRobotEpisode) => Promise<Record<string, unknown>[]>;

  if (isV3) {
    const adapter = new V30Adapter();
    info = await adapter.loadInfo(root);
    episodes = await adapter.loadEpisodes({ root, info });
    resolveKey = featureKeyMap(info);

    // For v3.0, read only the episode's rows from the shard using frameRange.
    let shardCache = new Map<string, Record<string, unknown>[]>();
    readEpisodeRows = async (ep: LeRobotEpisode) => {
      const key = ep.dataShard
        ? `${ep.dataShard.chunkIndex}/${ep.dataShard.fileIndex}`
        : `ep${ep.episodeIndex}`;
      let rows = shardCache.get(key);
      if (!rows) {
        const dataPath = resolveV3DataPath(root, info, ep);
        if (!dataPath || !(await exists(dataPath))) return [];
        const { parquetReadObjects, asyncBufferFromFile } = await getHyparquet();
        const buffer = await asyncBufferFromFile(dataPath);
        rows = (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
        shardCache.set(key, rows!);
      }
      if (ep.frameRange) return (rows ?? []).slice(ep.frameRange[0], ep.frameRange[1]);
      return (rows ?? []).filter((r) => Number(r.episode_index) === ep.episodeIndex);
    };
  } else {
    const adapter = new V21Adapter();
    info = await adapter.loadInfo(root);
    episodes = await adapter.loadEpisodes({ root, info });
    resolveKey = featureKeyMap(info);
    readEpisodeRows = async (ep: LeRobotEpisode) => {
      const dataPath = await adapter.resolveDataFile({ root, info }, ep);
      if (!dataPath || !(await exists(dataPath))) return [];
      const { parquetReadObjects, asyncBufferFromFile } = await getHyparquet();
      const buffer = await asyncBufferFromFile(dataPath);
      return (await parquetReadObjects({ file: buffer })) as Record<string, unknown>[];
    };
  }

  if (episodes.length === 0) throw new Error("No episodes found.");

  const epStatsRecords: Record<string, unknown>[] = [];
  const globalAcc = new StatsAccumulator();
  const isV3Stats = isV3; // v3.0 writes stats.json directly, not episodes_stats.jsonl

  for (let i = 0; i < episodes.length; i++) {
    const ep = episodes[i];
    const rows = await readEpisodeRows(ep);
    if (rows.length === 0) { onProgress({ done: i + 1, total: episodes.length }); continue; }
    const clean = rows.map(sanitizeRow);

    const epAcc = new StatsAccumulator();
    epAcc.ingest(clean);
    globalAcc.ingest(clean);
    epStatsRecords.push({
      episode_index: ep.episodeIndex,
      stats: epAcc.toPerEpisode(resolveKey),
    });
    onProgress({ done: i + 1, total: episodes.length });
  }

  // Process video features (requires ffmpeg).
  const videoKeys = Object.keys(info.features).filter(
    (k) => info.features[k]?.dtype === "video",
  );
  if (videoKeys.length > 0 && !(await ffmpegAvailable())) {
    warnings.push(
      "ffmpeg is not installed or not in PATH. " +
      "Video/image feature statistics will be skipped. " +
      "Install ffmpeg to include per-channel RGB stats (e.g. `sudo apt install ffmpeg`)."
    );
  }
  for (const vk of videoKeys) {
    onProgress({ done: 0, total: 0 }); // signal video phase
    const vStats = await computeVideoFeatureStats(root, vk, (p) => {
      onProgress({ done: p.done, total: p.total });
    });
    if (vStats) {
      for (const rec of epStatsRecords) {
        (rec as Record<string, unknown>)[vk] = vStats;
      }
      (globalAcc as any)._videoStats = (globalAcc as any)._videoStats ?? {};
      (globalAcc as any)._videoStats[vk] = vStats;
    }
  }

  // Merge video stats into per-episode records (under "stats" key).
  if ((globalAcc as any)._videoStats) {
    for (const rec of epStatsRecords) {
      const s = (rec as Record<string, unknown>).stats as Record<string, unknown>;
      if (s) Object.assign(s, (globalAcc as any)._videoStats);
    }
  }
  if (isV3) {
    // v3.0: write global stats.json only (no per-episode stats file).
    const globalStats = globalAcc.toPerEpisode(resolveKey);
    if ((globalAcc as any)._videoStats) Object.assign(globalStats, (globalAcc as any)._videoStats);
    // floatify to prevent Python's json.load from inferring int64 for
    // whole-number values, which would cause Arrow type conflicts.
    const raw = JSON.stringify(globalStats, null, 2);
    await fs.writeFile(
      path.join(root, "meta", "stats.json"),
      floatifyArraysInJson(raw),
      "utf8",
    );
  } else {
    await writeStatsJsonl(path.join(root, "meta", "episodes_stats.jsonl"), epStatsRecords);
  }
  return warnings;
}

function resolveV3DataPath(
  root: string, info: LeRobotInfo, episode: LeRobotEpisode,
): string | undefined {
  if (!episode.dataShard) return undefined;
  const filled = buildDataPath({
    template: info.dataPath ?? "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    chunkIndex: episode.dataShard.chunkIndex,
    fileIndex: episode.dataShard.fileIndex,
    episodeIndex: episode.episodeIndex,
  });
  return path.join(root, filled);
}

// ---- stats accumulator (Welford online algorithm) ----

class StatsAccumulator {
  private mins = new Map<string, number[]>();
  private maxs = new Map<string, number[]>();
  private means = new Map<string, number[]>();
  private m2s = new Map<string, number[]>();
  private counts = new Map<string, number>();
  // Per-dimension value arrays for quantile computation.
  private values = new Map<string, number[][]>();
  private valCounts = new Map<string, number>();
  private static readonly MAX_SAMPLES = 20000;

  ingest(rows: Record<string, unknown>[]): void {
    for (const row of rows) {
      for (const [key, value] of Object.entries(row)) {
        if (value === null || value === undefined) continue;
        const arr = Array.isArray(value) ? (value as number[]) : [value as number];
        if (arr.some((v) => typeof v !== "number" || !Number.isFinite(v))) continue;
        if (!this.counts.has(key)) this.init(key, arr.length);
        const count = this.counts.get(key)! + 1;
        this.counts.set(key, count);
        const mins = this.mins.get(key)!;
        const maxs = this.maxs.get(key)!;
        const means = this.means.get(key)!;
        const m2s = this.m2s.get(key)!;
        const vals = this.values.get(key)!;
        const vc = this.valCounts.get(key)! + 1;
        this.valCounts.set(key, vc);
        for (let j = 0; j < arr.length; j++) {
          const x = arr[j];
          // Reservoir sampling for quantile values.
          if (vals[j].length < StatsAccumulator.MAX_SAMPLES) {
            vals[j].push(x);
          } else {
            const idx = Math.floor(Math.random() * vc);
            if (idx < StatsAccumulator.MAX_SAMPLES) vals[j][idx] = x;
          }
          if (x < mins[j]) mins[j] = x;
          if (x > maxs[j]) maxs[j] = x;
          const delta = x - means[j];
          means[j] += delta / count;
          const delta2 = x - means[j];
          m2s[j] += delta * delta2;
        }
      }
    }
  }

  private init(key: string, n: number): void {
    this.mins.set(key, new Array(n).fill(Infinity));
    this.maxs.set(key, new Array(n).fill(-Infinity));
    this.means.set(key, new Array(n).fill(0));
    this.m2s.set(key, new Array(n).fill(0));
    this.counts.set(key, 0);
    this.valCounts.set(key, 0);
    this.values.set(key, Array.from({ length: n }, () => []));
  }

  toPerEpisode(resolveKey: (pk: string) => string): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const pk of this.counts.keys()) {
      const count = this.counts.get(pk)!;
      const vals = this.values.get(pk)!;
      // Pre-sort each dimension once, then compute all quantiles from the sorted arrays.
      const sorted = vals.map((v) => v.slice().sort((a, b) => a - b));
      const q01 = sorted.map((s) => quantileFromSorted(s, 0.01));
      const q10 = sorted.map((s) => quantileFromSorted(s, 0.10));
      const q50 = sorted.map((s) => quantileFromSorted(s, 0.50));
      const q90 = sorted.map((s) => quantileFromSorted(s, 0.90));
      const q99 = sorted.map((s) => quantileFromSorted(s, 0.99));
      out[resolveKey(pk)] = {
        min: this.mins.get(pk),
        max: this.maxs.get(pk),
        mean: this.means.get(pk)!,
        std: this.m2s.get(pk)!.map((m2) => Math.sqrt(m2 / count)),
        q01,
        q10,
        q50,
        q90,
        q99,
        count: [count],
      };
    }
    return out;
  }
}

/** Return the q-th quantile of an already-sorted array (linear interpolation). */
function quantileFromSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

function featureKeyMap(info: LeRobotInfo): (pk: string) => string {
  const featureKeys = Object.keys(info.features);
  return (pk: string): string => {
    if (featureKeys.includes(pk)) return pk;
    const lower = pk.toLowerCase();
    return featureKeys.find((fk) => fk.toLowerCase() === lower) ?? pk;
  };
}

// ---- helpers ----

function sanitizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (typeof value === "bigint") out[key] = Number(value);
    else if (Array.isArray(value)) out[key] = value.map((v) => (typeof v === "bigint" ? Number(v) : v));
    else out[key] = value;
  }
  return out;
}
