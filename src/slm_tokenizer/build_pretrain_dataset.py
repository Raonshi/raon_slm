from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardMetadata:
    shard_id: int
    path: str
    dtype: str
    token_count: int
    document_count: int
    source_start_line: int
    source_end_line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize normalized text and build sharded pretraining dataset files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/tokenizer/train.txt"),
        help="Normalized UTF-8 training text, one document per line.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/tokenizer/ko_spm.model"),
        help="SentencePiece model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/pretrain"),
        help="Directory for token shards and metadata.",
    )
    parser.add_argument(
        "--shard-token-count",
        type=int,
        default=1_000_000,
        help="Target number of tokens per shard.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend the tokenizer BOS id to each document.",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not append the tokenizer EOS id to each document.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_dtype(vocab_size: int) -> np.dtype[Any]:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")


def write_manifest(path: Path, shards: list[ShardMetadata]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for shard in shards:
            output_file.write(json.dumps(asdict(shard), ensure_ascii=False, sort_keys=True) + "\n")


def flush_shard(
    *,
    output_dir: Path,
    shard_id: int,
    tokens: list[int],
    dtype: np.dtype[Any],
    document_count: int,
    source_start_line: int,
    source_end_line: int,
) -> ShardMetadata:
    shard_name = f"train-{shard_id:05d}.npy"
    shard_path = output_dir / shard_name
    array = np.asarray(tokens, dtype=dtype)
    np.save(shard_path, array)

    return ShardMetadata(
        shard_id=shard_id,
        path=shard_name,
        dtype=str(dtype),
        token_count=int(array.size),
        document_count=document_count,
        source_start_line=source_start_line,
        source_end_line=source_end_line,
    )


def build_pretrain_dataset(
    *,
    input_path: Path,
    model_path: Path,
    output_dir: Path,
    shard_token_count: int,
    add_bos: bool,
    add_eos: bool,
) -> dict[str, Any]:
    if shard_token_count <= 0:
        raise ValueError("--shard-token-count must be positive.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input text does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model does not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    dtype = token_dtype(processor.vocab_size())
    bos_id = processor.bos_id()
    eos_id = processor.eos_id()

    shards: list[ShardMetadata] = []
    shard_tokens: list[int] = []
    shard_documents = 0
    shard_start_line = 1
    shard_end_line = 0
    token_count = 0
    document_count = 0

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue

            ids = processor.encode(text, out_type=int)
            if add_bos and bos_id >= 0:
                ids.insert(0, bos_id)
            if add_eos and eos_id >= 0:
                ids.append(eos_id)

            if not shard_tokens:
                shard_start_line = line_number
            shard_end_line = line_number

            shard_tokens.extend(ids)
            shard_documents += 1
            document_count += 1
            token_count += len(ids)

            if len(shard_tokens) >= shard_token_count:
                shards.append(
                    flush_shard(
                        output_dir=output_dir,
                        shard_id=len(shards),
                        tokens=shard_tokens,
                        dtype=dtype,
                        document_count=shard_documents,
                        source_start_line=shard_start_line,
                        source_end_line=line_number,
                    )
                )
                LOGGER.info(
                    "Wrote shard %s with %s tokens",
                    shards[-1].path,
                    shards[-1].token_count,
                )
                shard_tokens = []
                shard_documents = 0
                shard_end_line = 0

    if shard_tokens:
        shards.append(
            flush_shard(
                output_dir=output_dir,
                shard_id=len(shards),
                tokens=shard_tokens,
                dtype=dtype,
                document_count=shard_documents,
                source_start_line=shard_start_line,
                source_end_line=shard_end_line,
            )
        )
        LOGGER.info("Wrote shard %s with %s tokens", shards[-1].path, shards[-1].token_count)

    if document_count == 0:
        raise RuntimeError("No non-empty documents were found in the input text.")

    stats = {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "tokenizer_model_path": str(model_path),
        "tokenizer_model_sha256": sha256_file(model_path),
        "vocab_size": processor.vocab_size(),
        "dtype": str(dtype),
        "add_bos": add_bos,
        "add_eos": add_eos,
        "document_count": document_count,
        "token_count": token_count,
        "average_tokens_per_document": token_count / document_count,
        "shard_count": len(shards),
        "shard_token_count": shard_token_count,
        "special_token_ids": {
            "pad": processor.pad_id(),
            "unk": processor.unk_id(),
            "bos": bos_id,
            "eos": eos_id,
        },
    }

    write_manifest(output_dir / "manifest.jsonl", shards)
    write_json(output_dir / "stats.json", stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    stats = build_pretrain_dataset(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.output_dir,
        shard_token_count=args.shard_token_count,
        add_bos=args.add_bos,
        add_eos=not args.no_eos,
    )
    LOGGER.info(
        "Built %s shards with %s tokens from %s documents",
        stats["shard_count"],
        stats["token_count"],
        stats["document_count"],
    )


if __name__ == "__main__":
    main()
