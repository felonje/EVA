"""AffectiveState — 7D continuous emotional state.

Dimensions: valence, arousal, dominance, novelty_feeling, social, boredom, intrinsic_drive.
Updated via exponential moving averages from environmental signals.
Circuit breakers prevent emotional extremes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class AffectiveState:
    """Seven-dimensional continuous affective state.

    Dimensions:
        valence [-1, 1]: negative to positive feeling
        arousal [0, 1]: calm to excited
        dominance [0, 1]: submissive to dominant
        novelty_feeling [0, 1]: familiar to novel
        social [0, 1]: isolated to connected
        boredom [0, 1]: low to high boredom
        intrinsic_drive [0, 1]: low to high motivation to act/think

    All initialized to neutral values.
    """

    def __init__(self, ema_rate: float = 0.1) -> None:
        self.valence: float = 0.0
        self.arousal: float = 0.5
        self.dominance: float = 0.5
        self.novelty_feeling: float = 0.5
        self.social: float = 0.5
        self.boredom: float = 0.0
        self.intrinsic_drive: float = 0.5
        self._ema_rate = ema_rate

    def update(
        self,
        prediction_success: float,
        prediction_error: float,
        action_success: float,
        caregiver_recency: float,
        caregiver_contingency: float,
        novelty_signal: float = 0.0,
    ) -> None:
        """Update all affect dimensions from environmental signals.

        Args:
            prediction_success: How well predictions matched reality [0, 1].
            prediction_error: Absolute prediction error magnitude.
            action_success: Ratio of successful actions [0, 1].
            caregiver_recency: How recently caregiver interacted [0, 1].
            caregiver_contingency: Quality of caregiver response [0, 1].
            novelty_signal: New information discovered [0, 1].
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
        self.novelty_feeling = (
            self.novelty_feeling * (1 - r) + novelty_signal * r
        )

        # Boredom: increases when novelty is low, decreases when high
        # If novelty is low (< 0.2), boredom grows.
        target_boredom = 1.0 - novelty_signal
        self.boredom = self.boredom * (1 - r) + target_boredom * r

        # Intrinsic Drive: Sum of curiosity (novelty) and the need to reduce boredom
        # High boredom + High curiosity potential = High drive to act/think
        target_drive = (self.boredom + novelty_signal) / 2.0
        self.intrinsic_drive = self.intrinsic_drive * (1 - r) + target_drive * r

        # Social: caregiver recency * contingency
        target_social = caregiver_recency * caregiver_contingency
        self.social = self.social * (1 - r) + target_social * r

    def apply_circuit_breakers(self, config: Any) -> None:
        """Apply safety limits to prevent emotional extremes."""
        valence_floor = getattr(config, "valence_floor", -0.8)
        arousal_ceiling = getattr(config, "arousal_ceiling", 0.95)

        self.valence = max(valence_floor, min(1.0, self.valence))
        self.arousal = max(0.0, min(arousal_ceiling, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.novelty_feeling = max(0.0, min(1.0, self.novelty_feeling))
        self.social = max(0.0, min(1.0, self.social))
        self.boredom = max(0.0, min(1.0, self.boredom))
        self.intrinsic_drive = max(0.0, min(1.0, self.intrinsic_drive))

    def get_vector(self) -> np.ndarray:
        """Return affect as a numpy vector."""
        return np.array([
            self.valence,
            self.arousal,
            self.dominance,
            self.novelty_feeling,
            self.social,
            self.boredom,
            self.intrinsic_drive,
        ])

    def to_dict(self) -> dict[str, float]:
        """Return affect dimensions as a dictionary for logging."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "novelty_feeling": self.novelty_feeling,
            "social": self.social,
            "boredom": self.boredom,
            "intrinsic_drive": self.intrinsic_drive,
        }
