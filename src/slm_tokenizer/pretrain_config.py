from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    backend: str = "torch"
    dataset_dir: Path = Path("artifacts/datasets/pretrain")
    output_dir: Path = Path("artifacts/checkpoints/slm-10m")
    block_size: int = 512
    batch_size: int = 16
    max_steps: int = 1_000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    val_fraction: float = 0.05
    seed: int = 1337
    device: str = "auto"
