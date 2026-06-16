from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import sentencepiece as spm
from Korpora import Korpora

LOGGER = logging.getLogger(__name__)


DEFAULT_CORPORA = ("kcbert", "kowikitext", "korean_petitions")
WHITESPACE_RE = re.compile(r"\s+")
KOWIKITEXT_TRAIN_FILE = "kowikitext/kowikitext_20200920.train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Korean SentencePiece tokenizer with Korpora corpora."
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=list(DEFAULT_CORPORA),
        help=(
            "Korpora corpus names to use. "
            "Examples: kcbert kowikitext namuwikitext korean_petitions"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/korpora"),
        help="Directory where Korpora downloads raw corpora.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Directory for the normalized corpus and SentencePiece model.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16_000,
        help="SentencePiece vocabulary size.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200_000,
        help="Maximum normalized lines to write before training. Use 0 for no limit.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=10,
        help="Skip lines shorter than this after normalization.",
    )
    parser.add_argument(
        "--model-prefix",
        default="ko_spm",
        help="Output model prefix. Produces <prefix>.model and <prefix>.vocab.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Recommended high coverage for Korean.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\u200b", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def iter_samples(corpus: object) -> Iterator[object]:
    for split_name in ("train", "dev", "test"):
        split = getattr(corpus, split_name, None)
        if split is not None:
            yield from split


def iter_texts(corpus: object) -> Iterator[str]:
    """Yield text fields from Korpora corpus objects without depending on one schema."""
    for sample in iter_samples(corpus):
        if isinstance(sample, str):
            yield sample
            continue

        for attr in ("text", "sentence", "sentences", "document", "paragraph", "title"):
            value = getattr(sample, attr, None)
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                yield from (item for item in value if isinstance(item, str))


def iter_plain_text_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as input_file:
        yield from input_file


def iter_local_kowikitext(data_dir: Path) -> Iterator[str] | None:
    path = data_dir / KOWIKITEXT_TRAIN_FILE
    if not path.exists():
        return None

    LOGGER.info("Streaming local kowikitext file without loading it into memory: %s", path)
    return iter_plain_text_lines(path)


def iter_corpus_texts(corpus_name: str, data_dir: Path) -> Iterator[str]:
    if corpus_name == "kowikitext":
        local_texts = iter_local_kowikitext(data_dir)
        if local_texts is not None:
            yield from local_texts
            return

    LOGGER.info("Loading Korpora corpus: %s", corpus_name)
    corpus = Korpora.load(corpus_name, root_dir=str(data_dir))
    yield from iter_texts(corpus)


def write_training_corpus(
    corpora: Iterable[str],
    data_dir: Path,
    output_path: Path,
    max_lines: int,
    min_chars: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = None if max_lines == 0 else max_lines
    line_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for corpus_name in corpora:
            for text in iter_corpus_texts(corpus_name, data_dir):
                normalized = normalize_text(text)
                if len(normalized) < min_chars:
                    continue

                output_file.write(normalized + "\n")
                line_count += 1

                if limit is not None and line_count >= limit:
                    return line_count

    return line_count


def train_sentencepiece(
    input_path: Path,
    output_dir: Path,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
) -> Path:
    model_path_prefix = output_dir / model_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_path_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<mask>", "<sep>", "<cls>"],
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )

    return model_path_prefix.with_suffix(".model")


def preview_tokenizer(model_path: Path) -> None:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    examples = [
        "안녕하세요. 작은 한국어 언어모델을 위한 토크나이저를 학습합니다.",
        "토크나이저 품질은 데이터 정제와 어휘 크기에 크게 영향을 받습니다.",
    ]

    for example in examples:
        pieces = processor.encode(example, out_type=str)
        ids = processor.encode(example, out_type=int)
        LOGGER.info("Text: %s", example)
        LOGGER.info("Pieces: %s", pieces)
        LOGGER.info("Ids: %s", ids)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    training_corpus_path = args.output_dir / "train.txt"
    line_count = write_training_corpus(
        corpora=args.corpora,
        data_dir=args.data_dir,
        output_path=training_corpus_path,
        max_lines=args.max_lines,
        min_chars=args.min_chars,
    )

    if line_count == 0:
        raise RuntimeError("No training lines were written. Try different corpora or min-chars.")

    LOGGER.info("Wrote %s normalized lines to %s", line_count, training_corpus_path)
    model_path = train_sentencepiece(
        input_path=training_corpus_path,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )
    LOGGER.info("Saved tokenizer model to %s", model_path)
    preview_tokenizer(model_path)


if __name__ == "__main__":
    main()
