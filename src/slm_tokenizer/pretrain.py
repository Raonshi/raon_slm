from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from slm_tokenizer.backends import BackendUnavailableError, create_backend
from slm_tokenizer.pretrain_config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a 10M Korean causal language model.")
    parser.add_argument(
        "--backend",
        default=TrainConfig.backend,
        choices=("torch", "mlx"),
        help="Training backend. Use torch for portability or mlx for Apple Silicon.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=TrainConfig.dataset_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=TrainConfig.eval_iters)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--device",
        default=TrainConfig.device,
        choices=("auto", "cpu", "cuda", "mps", "gpu"),
        help="Torch: auto/cpu/cuda/mps. MLX: auto/cpu/mps/gpu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build one batch and run one forward/backward step without saving a checkpoint.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        backend=args.backend,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
    backend = create_backend(config)
    if args.dry_run:
        backend.dry_run()
    else:
        backend.train()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    try:
        train(args)
    except BackendUnavailableError as error:
        logging.getLogger(__name__).error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
