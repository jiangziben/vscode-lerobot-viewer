#!/usr/bin/env python3
"""Fix the global `index` column in a LeRobot v2.1 or v3.0 dataset.

After merging datasets or deleting episodes, the `index` column (global
frame counter) may become non-contiguous or have duplicates.  This script
rewrites parquet files with correct, monotonically increasing index values
starting from 0.

Usage:
    python fix_global_index.py /path/to/dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def fix_v21(root: Path) -> int:
    """Fix per-episode parquet files (v2.x layout)."""
    import pandas as pd

    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"{episodes_path} not found")

    episodes = pd.read_json(episodes_path, lines=True)
    episodes = episodes.sort_values("episode_index")
    data_dir = root / "data"

    fixed = 0
    cum_frames = 0

    for _, ep in tqdm(episodes.iterrows(), total=len(episodes), desc="v2.1"):
        ep_idx = int(ep["episode_index"])
        chunk = f"{ep_idx // 1000:03d}"
        fname = f"episode_{ep_idx:06d}.parquet"
        parquet_path = data_dir / f"chunk-{chunk}" / fname

        if not parquet_path.exists():
            candidates = list(data_dir.glob(f"*/{fname}"))
            if not candidates:
                tqdm.write(f"  [WARN] ep {ep_idx}: not found, skipping")
                cum_frames += int(ep.get("length", 0))
                continue
            parquet_path = candidates[0]

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "index" not in df.columns:
            tqdm.write(f"  [WARN] ep {ep_idx}: no 'index' column, skipping")
            cum_frames += len(df)
            continue

        if df["index"].iloc[0] == cum_frames and len(df) == int(df["index"].iloc[-1] - df["index"].iloc[0] + 1):
            cum_frames += len(df)
            continue

        df["index"] = range(cum_frames, cum_frames + len(df))
        cum_frames += len(df)

        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        shutil.move(str(tmp_path), str(parquet_path))
        fixed += 1

    return fixed


def fix_v30(root: Path) -> int:
    """Fix sharded parquet files (v3.0 layout)."""
    data_dir = root / "data"
    shard_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found under {data_dir}")

    fixed = 0
    cum_frames = 0

    for shard_path in tqdm(shard_files, desc="v3.0"):
        table = pq.read_table(shard_path)
        df = table.to_pandas()

        if "index" not in df.columns:
            tqdm.write(f"  [WARN] {shard_path.name}: no 'index' column, skipping")
            cum_frames += len(df)
            continue

        if df["index"].iloc[0] == cum_frames and len(df) == int(df["index"].iloc[-1] - df["index"].iloc[0] + 1):
            cum_frames += len(df)
            continue

        df["index"] = range(cum_frames, cum_frames + len(df))
        cum_frames += len(df)

        tmp_path = shard_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        shutil.move(str(tmp_path), str(shard_path))
        fixed += 1

    return fixed


def detect_version(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    return info.get("codebase_version", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix global index column in LeRobot dataset")
    parser.add_argument("root", type=Path, help="Path to the dataset root directory")
    args = parser.parse_args()

    version = detect_version(args.root)
    print(f"Detected: {version}")

    if version in ("v2.0", "v2.1"):
        fixed = fix_v21(args.root)
    elif version == "v3.0":
        fixed = fix_v30(args.root)
    else:
        raise SystemExit(f"Unsupported version: {version}")

    print(f"\nDone. Fixed {fixed} parquet files.")


if __name__ == "__main__":
    main()
