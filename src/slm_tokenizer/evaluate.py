from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slm_tokenizer.pretrain_config import TrainConfig
from slm_tokenizer.pretrain_data import load_token_stream, split_tokens
from slm_tokenizer.torch_runtime import (
    load_checkpoint,
    load_model_from_checkpoint,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained Korean causal LM.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=TrainConfig.output_dir / "torch" / "best.pt",
        help="Path to a PyTorch checkpoint produced by slm_tokenizer.pretrain.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/slm-10m/torch/best.json"),
        help="Where to write evaluation metrics as JSON.",
    )
    return parser.parse_args()


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.size - block_size - 1, (batch_size,))
    x = np.stack([data[index : index + block_size] for index in starts.tolist()])
    y = np.stack([data[index + 1 : index + block_size + 1] for index in starts.tolist()])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    output: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = get_batch(data, batch_size, model.config.block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Loss was not computed.")
            losses[index] = loss.item()
        output[split] = losses.mean().item()
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    train_config = checkpoint["train_config"]
    model = load_model_from_checkpoint(checkpoint, device)
    model_config = model.config
    dataset_dir = args.dataset_dir or Path(train_config["dataset_dir"])
    batch_size = args.batch_size or int(train_config["batch_size"])
    val_fraction = args.val_fraction or float(train_config["val_fraction"])
    seed = args.seed if args.seed is not None else int(train_config["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens, dataset_stats = load_token_stream(dataset_dir)
    train_data, val_data = split_tokens(tokens, val_fraction, model_config.block_size)

    losses = estimate_loss(model, train_data, val_data, batch_size, args.eval_iters, device)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_best_val_loss": float(checkpoint["best_val_loss"]),
        "device": str(device),
        "eval_iters": args.eval_iters,
        "batch_size": batch_size,
        "block_size": model_config.block_size,
        "dataset_dir": str(dataset_dir),
        "dataset_tokens": int(tokens.size),
        "dataset_stats": dataset_stats,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "train_perplexity": math.exp(losses["train"]),
        "val_perplexity": math.exp(losses["val"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")

    LOGGER.info(
        "train_loss=%.4f train_ppl=%.2f val_loss=%.4f val_ppl=%.2f",
        metrics["train_loss"],
        metrics["train_perplexity"],
        metrics["val_loss"],
        metrics["val_perplexity"],
    )
    LOGGER.info("Wrote evaluation metrics to %s", args.output)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(parse_args())


if __name__ == "__main__":
    main()
