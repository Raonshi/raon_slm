from __future__ import annotations

from typing import Protocol


class BackendUnavailableError(RuntimeError):
    """Raised when a selected backend cannot run in the current environment."""


class PretrainBackend(Protocol):
    name: str

    def dry_run(self) -> None:
        """Run one forward/backward step without writing checkpoints."""

    def train(self) -> None:
        """Run the configured pretraining loop."""
