from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_token_stream(dataset_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    stats_path = dataset_dir / "stats.json"
    manifest_path = dataset_dir / "manifest.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset stats: {stats_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    stats = read_json(stats_path)
    shard_paths: list[Path] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if line.strip():
                shard_paths.append(dataset_dir / json.loads(line)["path"])
    if not shard_paths:
        raise RuntimeError(f"No shards listed in {manifest_path}")

    arrays = [np.load(path, mmap_mode="r") for path in shard_paths]
    tokens = np.concatenate(arrays).astype(np.int64, copy=False)
    return tokens, stats


def split_tokens(
    tokens: np.ndarray,
    val_fraction: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5.")
    if tokens.size < 10 * block_size:
        raise ValueError("Dataset is too small for the requested block size.")

    val_size = max(int(tokens.size * val_fraction), block_size + 1)
    train = tokens[:-val_size]
    val = tokens[-val_size:]
    if train.size <= block_size or val.size <= block_size:
        raise ValueError("Train/validation split is too small for the requested block size.")
    return train, val
