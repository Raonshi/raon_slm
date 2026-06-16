# SLM

Korpora로 한국어 말뭉치를 내려받고 SentencePiece unigram tokenizer를 학습하는
작은 한국어 causal language model 예제 프로젝트입니다.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Apple Silicon에서 MLX backend를 함께 사용하려면 optional dependency를 설치합니다.

```bash
pip install -e '.[mlx]'
```

## Train

기본값은 `kcbert`, `kowikitext`, `korean_petitions`를 사용하고 최대 20만 줄만
정규화해서 학습합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.train_sentencepiece
```

더 작게 실험하려면 줄 수와 vocab size를 낮추면 됩니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.train_sentencepiece \
  --corpora kowikitext \
  --max-lines 50000 \
  --vocab-size 8000
```

학습이 끝나면 다음 파일이 생성됩니다.

```text
artifacts/tokenizer/train.txt
artifacts/tokenizer/ko_spm.model
artifacts/tokenizer/ko_spm.vocab
```

## Build Pretraining Dataset

토크나이저 학습이 끝난 뒤에는 정규화된 `train.txt`를 토큰 ID로 변환하고,
학습 로더가 읽기 쉬운 shard로 나눕니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.build_pretrain_dataset
```

기본 출력은 다음과 같습니다.

```text
artifacts/datasets/pretrain/train-00000.npy
artifacts/datasets/pretrain/manifest.jsonl
artifacts/datasets/pretrain/stats.json
```

작은 shard로 파이프라인을 검증하려면 다음처럼 실행합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.build_pretrain_dataset \
  --shard-token-count 100000
```

`stats.json`에는 입력 파일과 토크나이저 모델의 SHA-256, 문서 수, 토큰 수,
평균 문서 토큰 길이, 특수 토큰 ID가 기록됩니다. `manifest.jsonl`은 각 shard의
토큰 수, 문서 수, 원본 줄 범위를 담습니다.

## Pretrain 10M Model

현재 기본 모델은 약 10M 파라미터의 GPT-style causal LM입니다.

```text
vocab_size: dataset stats에서 자동 로드
context_length: 512
n_layers: 4
n_heads: 6
d_model: 384
dropout: 0.1
```

프리트레이닝은 Strategy + Factory 구조로 backend를 선택합니다.

```text
--backend torch  # PyTorch backend, 기본값
--backend mlx    # Apple Silicon용 MLX backend
```

먼저 한 배치 forward/backward만 확인하려면 다음처럼 실행합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.pretrain --backend torch --dry-run
```

MLX backend는 Apple Silicon의 일반 터미널 환경에서 실행합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.pretrain --backend mlx --dry-run
```

짧은 실험 학습은 다음처럼 실행합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.pretrain \
  --backend torch \
  --batch-size 16 \
  --block-size 512 \
  --max-steps 1000
```

체크포인트는 기본적으로 다음 위치에 저장됩니다.

```text
artifacts/checkpoints/slm-10m/torch/best.pt
artifacts/checkpoints/slm-10m/torch/last.pt
artifacts/checkpoints/slm-10m/mlx/best/weights.npz
artifacts/checkpoints/slm-10m/mlx/best/metadata.json
```

PyTorch와 MLX는 tensor, optimizer, checkpoint 형식이 다르므로 모델과 학습 루프는
backend별 구현체가 소유합니다. CLI 설정, dataset shard, train/validation split,
logging 규칙만 공통으로 공유합니다.

현재 예제 데이터셋은 약 182만 토큰이므로 모델 품질을 기대하기보다는 학습 루프,
loss 감소, checkpoint 저장, 평가 파이프라인 검증에 초점을 둡니다.

## Evaluate

프리트레이닝 체크포인트를 고정한 뒤 train/validation loss와 perplexity를 계산합니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.evaluate \
  --checkpoint artifacts/checkpoints/slm-10m/torch/best.pt \
  --eval-iters 100
```

기본 출력은 다음 위치에 JSON으로 저장됩니다.

```text
artifacts/evaluations/slm-10m/torch/best.json
```

## Generate Samples

loss만으로는 한국어 생성 품질을 체감하기 어려우므로, 체크포인트와 SentencePiece
토크나이저를 연결해 prompt 기반 샘플을 생성할 수 있습니다.

```bash
PYTHONPATH=src python -m slm_tokenizer.generate \
  --checkpoint artifacts/checkpoints/slm-10m/torch/best.pt \
  --prompt "오늘의 인공지능 연구는" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50
```

`--checkpoint`를 `best.pt`와 `last.pt`로 바꿔 같은 prompt를 넣으면 loss 비교와 별개로
두 체크포인트의 한국어 생성 느낌을 빠르게 비교할 수 있습니다. `--prompt`를 생략하면
터미널에서 직접 입력받습니다.

## Notes

- Korpora 패키지의 라이선스와 각 말뭉치의 라이선스는 다를 수 있습니다. 모델을
  배포하거나 상업적으로 사용할 계획이면 corpus별 원 라이선스를 따로 확인하세요.
- 처음에는 `--max-lines 50000` 정도로 tokenizer 파이프라인을 검증한 뒤,
  데이터와 vocab size를 늘리는 흐름이 좋습니다.
- 한국어 tokenizer는 `character_coverage`를 높게 잡는 편이 안전합니다. 이 예제는
  `0.9995`를 기본값으로 사용합니다.
