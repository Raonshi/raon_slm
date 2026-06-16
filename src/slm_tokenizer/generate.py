from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from slm_tokenizer.pretrain_config import TrainConfig
from slm_tokenizer.torch_runtime import (
    load_checkpoint,
    load_model_from_checkpoint,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Korean samples from a pretrained LM.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=TrainConfig.output_dir / "torch" / "best.pt",
        help="Path to a PyTorch checkpoint produced by slm_tokenizer.pretrain.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help=(
            "SentencePiece model path. Defaults to checkpoint dataset metadata, "
            "then artifacts/tokenizer/ko_spm.model."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, the command asks for one on stdin.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Generation device.",
    )
    parser.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Stop printed output at the first generated EOS token if present.",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Log prompt and generated token ids for debugging.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")


def resolve_tokenizer_path(args: argparse.Namespace, checkpoint: dict[str, Any]) -> Path:
    if args.tokenizer is not None:
        return args.tokenizer

    dataset_stats = checkpoint.get("dataset_stats", {})
    tokenizer_path = dataset_stats.get("tokenizer_model_path")
    if tokenizer_path:
        return Path(tokenizer_path)

    return Path("artifacts/tokenizer/ko_spm.model")


def read_prompt(prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    return input("Prompt: ")


def truncate_at_generated_eos(
    token_ids: list[int],
    prompt_length: int,
    eos_id: int,
) -> list[int]:
    if eos_id < 0:
        return token_ids
    for index in range(prompt_length, len(token_ids)):
        if token_ids[index] == eos_id:
            return token_ids[:index]
    return token_ids


@torch.no_grad()
def generate(args: argparse.Namespace) -> str:
    validate_args(args)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer_path = resolve_tokenizer_path(args, checkpoint)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer model: {tokenizer_path}")

    prompt = read_prompt(args.prompt)
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    prompt_ids = processor.encode(prompt, out_type=int)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    seed = args.seed
    if seed is None:
        seed = int(checkpoint.get("train_config", {}).get("seed", TrainConfig.seed))
    torch.manual_seed(seed)

    model = load_model_from_checkpoint(checkpoint, device)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()

    if args.stop_at_eos:
        output_ids = truncate_at_generated_eos(
            output_ids,
            prompt_length=len(prompt_ids),
            eos_id=processor.eos_id(),
        )

    if args.show_token_ids:
        LOGGER.info("prompt_ids=%s", prompt_ids)
        LOGGER.info("output_ids=%s", output_ids)

    text = processor.decode(output_ids)
    LOGGER.info(
        "Generated %s new tokens from checkpoint step %s on %s",
        len(output_ids) - len(prompt_ids),
        checkpoint.get("step", "unknown"),
        device,
    )
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(generate(parse_args()))


if __name__ == "__main__":
    main()
