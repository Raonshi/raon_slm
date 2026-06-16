from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class MLXGPTConfig:
    vocab_size: int = 8_000
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class MLXBlock(nn.Module):
    def __init__(self, config: MLXGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiHeadAttention(
            dims=config.n_embd,
            num_heads=config.n_head,
            bias=config.bias,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.mlp_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        y = self.ln_1(x)
        y = self.attn(y, y, y, mask)
        x = x + self.attn_dropout(y)

        y = self.ln_2(x)
        y = self.c_proj(nn.gelu_approx(self.c_fc(y)))
        return x + self.mlp_dropout(y)


class MLXGPTLanguageModel(nn.Module):
    def __init__(self, config: MLXGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = [MLXBlock(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)

    def __call__(self, idx: mx.array) -> mx.array:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; block size is "
                f"{self.config.block_size}."
            )

        positions = mx.arange(seq_len)
        x = self.wte(idx) + self.wpe(positions)
        x = self.drop(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.wte.as_linear(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def parameter_count(self) -> int:
        def count(value: object) -> int:
            if isinstance(value, mx.array):
                return value.size
            if isinstance(value, dict):
                return sum(count(child) for child in value.values())
            if isinstance(value, list | tuple):
                return sum(count(child) for child in value)
            return 0

        return count(self.parameters())
