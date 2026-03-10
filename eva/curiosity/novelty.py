"""Novelty Module — count-based state novelty tracking.

Tracks how often EVA has visited similar states. Novel states
(rarely visited) produce high novelty scores. Familiar states
produce low scores. Uses hashing of quantized hidden states.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import torch


class NoveltyModule:
    """Tracks state novelty via count-based method.

    Maintains a dictionary of state hashes to visit counts.
    Novelty score = 1 / sqrt(visit_count + 1), so novel states
    score high and familiar states score low.
    """

    def __init__(self, n_bins: int = 16) -> None:
        self._visit_counts: dict[str, int] = defaultdict(int)
        self._n_bins = n_bins

    def compute(self, state_hash: str) -> float:
        """Compute novelty score for a given state.

        Args:
            state_hash: Hash string of the current state.

        Returns:
            Novelty score: 1.0 / sqrt(visit_count + 1).
        """
        self._visit_counts[state_hash] += 1
        count = self._visit_counts[state_hash]
        return 1.0 / math.sqrt(count + 1)

    def hash_state(self, hidden_state: torch.Tensor) -> str:
        """Quantize hidden state to a hash string.

        Discretizes the hidden state into bins and hashes the
        resulting bin vector.

        Args:
            hidden_state: Hidden state tensor from BabyBrain.

        Returns:
            Hash string representing the discretized state.
        """
        # Take the mean across sequence dimension if needed
        if hidden_state.dim() == 3:
            state = hidden_state.mean(dim=1).squeeze(0)
        elif hidden_state.dim() == 2:
            state = hidden_state.mean(dim=0)
        else:
            state = hidden_state

        # Normalize to [0, 1] range
        state = state.float()
        state_min = state.min()
        state_max = state.max()
        if state_max - state_min > 1e-8:
            state = (state - state_min) / (state_max - state_min)
        else:
            state = torch.zeros_like(state)

        # Discretize into bins
        bins = (state * (self._n_bins - 1)).long().clamp(0, self._n_bins - 1)
        bin_bytes = bins.cpu().numpy().tobytes()

        return hashlib.md5(bin_bytes).hexdigest()

    def reset(self) -> None:
        """Clear all visit counts."""
        self._visit_counts.clear()
