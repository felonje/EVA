"""AffectiveState — 5D continuous emotional state.

Dimensions: valence, arousal, dominance, novelty_feeling, social.
Updated via exponential moving averages from environmental signals.
Circuit breakers prevent emotional extremes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class AffectiveState:
    """Five-dimensional continuous affective state.

    Dimensions:
        valence [-1, 1]: negative to positive feeling
        arousal [0, 1]: calm to excited
        dominance [0, 1]: submissive to dominant
        novelty_feeling [0, 1]: familiar to novel
        social [0, 1]: isolated to connected

    All initialized to neutral values.
    """

    def __init__(self, ema_rate: float = 0.1) -> None:
        self.valence: float = 0.0
        self.arousal: float = 0.5
        self.dominance: float = 0.5
        self.novelty_feeling: float = 0.5
        self.social: float = 0.5
        self._ema_rate = ema_rate

    def update(
        self,
        prediction_success: float,
        prediction_error: float,
        action_success: float,
        caregiver_recency: float,
        caregiver_contingency: float,
    ) -> None:
        """Update all affect dimensions from environmental signals.

        Args:
            prediction_success: How well predictions matched reality [0, 1].
            prediction_error: Absolute prediction error magnitude.
            action_success: Ratio of successful actions [0, 1].
            caregiver_recency: How recently caregiver interacted [0, 1].
            caregiver_contingency: Quality of caregiver response [0, 1].
        """
        r = self._ema_rate

        # Valence: positive predictions -> positive valence
        target_valence = prediction_success * 2.0 - 1.0  # Map [0,1] to [-1,1]
        self.valence = self.valence * (1 - r) + target_valence * r

        # Arousal: big errors -> high arousal
        target_arousal = min(1.0, abs(prediction_error))
        self.arousal = self.arousal * (1 - r) + target_arousal * r

        # Dominance: action success -> dominance
        self.dominance = self.dominance * (1 - r) + action_success * r

        # Novelty feeling: driven by curiosity novelty signal
        # (caller passes this through prediction_error for now)
        target_novelty = min(1.0, prediction_error * 0.5)
        self.novelty_feeling = (
            self.novelty_feeling * (1 - r) + target_novelty * r
        )

        # Social: caregiver recency * contingency
        target_social = caregiver_recency * caregiver_contingency
        self.social = self.social * (1 - r) + target_social * r

    def apply_circuit_breakers(self, config: Any) -> None:
        """Apply safety limits to prevent emotional extremes.

        Args:
            config: Config section with valence_floor, arousal_ceiling, etc.
        """
        valence_floor = getattr(config, "valence_floor", -0.8)
        arousal_ceiling = getattr(config, "arousal_ceiling", 0.95)

        self.valence = max(valence_floor, min(1.0, self.valence))
        self.arousal = max(0.0, min(arousal_ceiling, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.novelty_feeling = max(0.0, min(1.0, self.novelty_feeling))
        self.social = max(0.0, min(1.0, self.social))

    def get_vector(self) -> np.ndarray:
        """Return affect as a numpy vector.

        Returns:
            Array of [valence, arousal, dominance, novelty_feeling, social].
        """
        return np.array([
            self.valence,
            self.arousal,
            self.dominance,
            self.novelty_feeling,
            self.social,
        ])

    def to_dict(self) -> dict[str, float]:
        """Return affect dimensions as a dictionary for logging."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "novelty_feeling": self.novelty_feeling,
            "social": self.social,
        }
