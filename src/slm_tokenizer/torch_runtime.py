from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from slm_tokenizer.model import GPTConfig, GPTLanguageModel


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("backend") != "torch":
        raise ValueError("This command currently supports PyTorch checkpoints only.")
    return checkpoint


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> GPTLanguageModel:
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
