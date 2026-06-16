from __future__ import annotations

from slm_tokenizer.backends.base import PretrainBackend
from slm_tokenizer.pretrain_config import TrainConfig


def create_backend(config: TrainConfig) -> PretrainBackend:
    if config.backend == "torch":
        from slm_tokenizer.backends.torch_backend import TorchPretrainBackend

        return TorchPretrainBackend(config)
    if config.backend == "mlx":
        from slm_tokenizer.backends.mlx_backend import MLXPretrainBackend

        return MLXPretrainBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
