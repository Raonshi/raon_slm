from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except RuntimeError as error:
    from slm_tokenizer.backends.base import BackendUnavailableError

    raise BackendUnavailableError(
        "MLX could not initialize. On macOS this usually means the current session "
        "cannot access a Metal device. Run the same command from a normal Apple "
        "Silicon terminal, or use '--backend torch'."
    ) from error
except ImportError as error:
    from slm_tokenizer.backends.base import BackendUnavailableError

    raise BackendUnavailableError(
        "The MLX backend requires the optional dependency 'mlx'. Install it with "
        "`pip install -e '.[mlx]'` or `pip install mlx`."
    ) from error

from slm_tokenizer.mlx_model import MLXGPTConfig, MLXGPTLanguageModel
from slm_tokenizer.pretrain_config import TrainConfig
from slm_tokenizer.pretrain_data import load_token_stream, split_tokens

LOGGER = logging.getLogger(__name__)


class MLXPretrainBackend:
    name = "mlx"

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.tokens, self.stats = load_token_stream(config.dataset_dir)
        self.train_data, self.val_data = split_tokens(
            self.tokens,
            config.val_fraction,
            config.block_size,
        )
        self.device = self.resolve_device(config.device)
        mx.random.seed(config.seed)
        np.random.seed(config.seed)
        mx.set_default_device(self.device)

        model_config = MLXGPTConfig(
            vocab_size=int(self.stats["vocab_size"]),
            block_size=config.block_size,
        )
        self.model = MLXGPTLanguageModel(model_config)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=[config.beta1, config.beta2],
            weight_decay=config.weight_decay,
        )
        mx.eval(self.model.parameters())

    @staticmethod
    def resolve_device(requested: str) -> mx.Device:
        if requested in {"auto", "mps", "gpu"}:
            return mx.gpu
        if requested == "cpu":
            return mx.cpu
        raise RuntimeError("MLX backend supports --device auto, cpu, mps, or gpu.")

    def get_batch(self, data: np.ndarray) -> tuple[mx.array, mx.array]:
        starts = np.random.randint(
            0,
            data.size - self.config.block_size - 1,
            size=(self.config.batch_size,),
        )
        x = np.stack([data[index : index + self.config.block_size] for index in starts])
        y = np.stack([data[index + 1 : index + self.config.block_size + 1] for index in starts])
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

    def loss_fn(self, x: mx.array, y: mx.array) -> mx.array:
        return self.model.loss(x, y)

    def train_step(self, x: mx.array, y: mx.array) -> float:
        loss_and_grad = nn.value_and_grad(self.model, self.loss_fn)
        loss, gradients = loss_and_grad(x, y)
        if self.config.grad_clip > 0:
            gradients, _ = optim.clip_grad_norm(gradients, self.config.grad_clip)
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return float(loss.item())

    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        for split, data in (("train", self.train_data), ("val", self.val_data)):
            losses: list[float] = []
            for _ in range(self.config.eval_iters):
                x, y = self.get_batch(data)
                loss = self.model.loss(x, y)
                mx.eval(loss)
                losses.append(float(loss.item()))
            out[split] = float(np.mean(losses))
        self.model.train()
        return out

    def save_checkpoint(self, path: Path, step: int, best_val_loss: float) -> None:
        path.mkdir(parents=True, exist_ok=True)
        weights_path = path / "weights.npz"
        metadata_path = path / "metadata.json"
        self.model.save_weights(str(weights_path))
        metadata = {
            "backend": self.name,
            "model_config": self.model.config.to_dict(),
            "train_config": {
                **asdict(self.config),
                "dataset_dir": str(self.config.dataset_dir),
                "output_dir": str(self.config.output_dir),
            },
            "dataset_stats": self.stats,
            "step": step,
            "best_val_loss": best_val_loss,
        }
        with metadata_path.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")

    def log_startup(self) -> None:
        LOGGER.info(
            "Loaded %s tokens: %s train, %s val",
            self.tokens.size,
            self.train_data.size,
            self.val_data.size,
        )
        LOGGER.info("Using backend/device: %s/%s", self.name, self.device)
        LOGGER.info("Model parameters: %.2fM", self.model.parameter_count() / 1_000_000)

    def dry_run(self) -> None:
        self.log_startup()
        x, y = self.get_batch(self.train_data)
        loss = self.train_step(x, y)
        LOGGER.info("Initial batch loss: %.4f", loss)
        LOGGER.info("Dry run completed.")

    def train(self) -> None:
        self.log_startup()
        x, y = self.get_batch(self.train_data)
        initial_loss = self.train_step(x, y)
        LOGGER.info("Initial batch loss: %.4f", initial_loss)

        best_val_loss = float("inf")
        last_time = time.time()
        for step in range(1, self.config.max_steps + 1):
            x, y = self.get_batch(self.train_data)
            self.train_step(x, y)

            if step % self.config.eval_interval == 0 or step == 1:
                losses = self.estimate_loss()
                elapsed = time.time() - last_time
                last_time = time.time()
                LOGGER.info(
                    "step %s/%s train_loss=%.4f val_loss=%.4f elapsed=%.1fs",
                    step,
                    self.config.max_steps,
                    losses["train"],
                    losses["val"],
                    elapsed,
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    self.save_checkpoint(
                        self.config.output_dir / self.name / "best",
                        step,
                        best_val_loss,
                    )

        self.save_checkpoint(
            self.config.output_dir / self.name / "last",
            self.config.max_steps,
            best_val_loss,
        )
        LOGGER.info("Saved checkpoints to %s", self.config.output_dir / self.name)
