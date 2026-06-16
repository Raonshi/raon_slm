from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from slm_tokenizer.model import GPTConfig, GPTLanguageModel
from slm_tokenizer.pretrain_config import TrainConfig
from slm_tokenizer.pretrain_data import load_token_stream, split_tokens

LOGGER = logging.getLogger(__name__)


class TorchPretrainBackend:
    name = "torch"

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.tokens, self.stats = load_token_stream(config.dataset_dir)
        self.train_data, self.val_data = split_tokens(
            self.tokens,
            config.val_fraction,
            config.block_size,
        )
        self.device = self.resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        model_config = GPTConfig(
            vocab_size=int(self.stats["vocab_size"]),
            block_size=config.block_size,
        )
        self.model = GPTLanguageModel(model_config).to(self.device)
        self.optimizer = self.configure_optimizer()

    @staticmethod
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

    def configure_optimizer(self) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dim() >= 2 and not name.endswith("wpe.weight"):
                decay.append(parameter)
            else:
                no_decay.append(parameter)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def get_batch(self, data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, data.size - self.config.block_size - 1, (self.config.batch_size,))
        x = np.stack(
            [data[index : index + self.config.block_size] for index in starts.tolist()]
        )
        y = np.stack(
            [data[index + 1 : index + self.config.block_size + 1] for index in starts.tolist()]
        )
        return (
            torch.from_numpy(x).long().to(self.device),
            torch.from_numpy(y).long().to(self.device),
        )

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        for split, data in (("train", self.train_data), ("val", self.val_data)):
            losses = torch.zeros(self.config.eval_iters)
            for index in range(self.config.eval_iters):
                x, y = self.get_batch(data)
                _, loss = self.model(x, y)
                if loss is None:
                    raise RuntimeError("Loss was not computed.")
                losses[index] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, path: Path, step: int, best_val_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": self.name,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_config": self.model.config.to_dict(),
                "train_config": {
                    **asdict(self.config),
                    "dataset_dir": str(self.config.dataset_dir),
                    "output_dir": str(self.config.output_dir),
                },
                "dataset_stats": self.stats,
                "step": step,
                "best_val_loss": best_val_loss,
            },
            path,
        )

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
        _, loss = self.model(x, y)
        if loss is None:
            raise RuntimeError("Loss was not computed.")
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)
        LOGGER.info("Initial batch loss: %.4f", loss.item())
        LOGGER.info("Dry run completed.")

    def train(self) -> None:
        self.log_startup()
        x, y = self.get_batch(self.train_data)
        _, loss = self.model(x, y)
        if loss is None:
            raise RuntimeError("Loss was not computed.")
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)
        LOGGER.info("Initial batch loss: %.4f", loss.item())

        best_val_loss = float("inf")
        last_time = time.time()
        for step in range(1, self.config.max_steps + 1):
            x, y = self.get_batch(self.train_data)
            _, loss = self.model(x, y)
            if loss is None:
                raise RuntimeError("Loss was not computed.")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

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
                        self.config.output_dir / self.name / "best.pt",
                        step,
                        best_val_loss,
                    )

        self.save_checkpoint(
            self.config.output_dir / self.name / "last.pt",
            self.config.max_steps,
            best_val_loss,
        )
        LOGGER.info("Saved checkpoints to %s", self.config.output_dir / self.name)
