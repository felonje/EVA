"""Empowerment Module — estimates available future options.

Measures how many distinct outcomes are reachable from the current
state. Higher empowerment means EVA has more control over its future.
Uses variance of recent outcome embeddings as a proxy.
"""

from __future__ import annotations

import torch


class EmpowermentModule:
    """Estimates how many future options are available from current state.

    Tracks recent (action, outcome) pairs and computes the diversity
    of outcomes. Higher diversity = more empowerment = EVA has more
    influence over what happens next.
    """

    def __init__(self, history_size: int = 50) -> None:
        self._history_size = history_size
        self._outcomes: list[torch.Tensor] = []

    def compute(self, recent_outcomes: list[torch.Tensor]) -> float:
        """Compute empowerment from diversity of recent outcomes.

        Args:
            recent_outcomes: List of outcome embedding tensors.

        Returns:
            Empowerment score (higher = more diverse outcomes).
        """
        if len(recent_outcomes) < 2:
            return 0.5  # Neutral when insufficient data

        # Stack outcomes and compute variance
        stacked = torch.stack(
            [o.float().flatten()[:256] for o in recent_outcomes[-self._history_size:]]
        )

        # Variance across outcomes — higher = more diverse
        variance = stacked.var(dim=0).mean().item()

        # Normalize to roughly [0, 1] range with sigmoid-like scaling
        empowerment = 2.0 / (1.0 + 1.0 / (variance + 1e-8)) - 1.0
        return max(0.0, min(1.0, empowerment))

    def add_outcome(self, outcome: torch.Tensor) -> None:
        """Add an outcome to the tracking history.

        Args:
            outcome: Outcome embedding tensor.
        """
        self._outcomes.append(outcome.detach())
        if len(self._outcomes) > self._history_size:
            self._outcomes.pop(0)

    def get_recent_outcomes(self) -> list[torch.Tensor]:
        """Return the list of recent outcome tensors."""
        return list(self._outcomes)
