"""Prediction Error Module — surprise from prediction vs actual outcome.

Computes how surprised EVA is by what actually happened versus what
it predicted. Maintains a running average to detect relative surprise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PredictionErrorModule:
    """Computes surprise from prediction vs actual outcome.

    Uses cross-entropy loss between predicted distribution and actual
    token. Maintains an exponential moving average to compute relative
    surprise (values > 1.0 mean more surprising than usual).
    """

    def __init__(self, ema_alpha: float = 0.01) -> None:
        self._ema_alpha = ema_alpha
        self._ema: float = 1.0  # Running average of errors
        self._initialized = False

    def compute(
        self, predicted_dist: torch.Tensor, actual_token: int
    ) -> float:
        """Compute prediction error as cross-entropy loss.

        Args:
            predicted_dist: Probability distribution over vocab,
                            shape (vocab_size,) or (1, vocab_size).
            actual_token: The actual token ID that occurred.

        Returns:
            Cross-entropy loss value (higher = more surprised).
        """
        if predicted_dist.dim() == 1:
            predicted_dist = predicted_dist.unsqueeze(0)

        target = torch.tensor([actual_token], device=predicted_dist.device)
        # Use log of probabilities for cross-entropy
        log_probs = torch.log(predicted_dist.float() + 1e-10)
        loss = F.nll_loss(log_probs, target)
        error = loss.item()

        # Update EMA
        if not self._initialized:
            self._ema = error
            self._initialized = True
        else:
            self._ema = (
                self._ema_alpha * error
                + (1 - self._ema_alpha) * self._ema
            )

        return error

    def get_relative_surprise(self, current_error: float) -> float:
        """Compute relative surprise: current error / running average.

        Args:
            current_error: The current prediction error.

        Returns:
            Relative surprise. Values > 1.0 = more surprising than usual.
        """
        if self._ema < 1e-10:
            return 1.0
        return current_error / self._ema

    def reset(self) -> None:
        """Reset the running average."""
        self._ema = 1.0
        self._initialized = False
