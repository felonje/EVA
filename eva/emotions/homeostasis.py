"""Homeostasis — basic drive system for EVA.

Three drives that increase when their needs are unmet:
- curiosity_hunger: need for intellectual stimulation
- rest_need: need for consolidation/downtime
- social_need: need for caregiver interaction
"""

from __future__ import annotations


class Homeostasis:
    """Three-drive homeostatic system.

    Each drive is a float [0, 1] where 1.0 = critically unmet.
    Drives increase when their needs aren't being met and decrease
    when they are satisfied.
    """

    def __init__(self) -> None:
        self.curiosity_hunger: float = 0.0
        self.rest_need: float = 0.0
        self.social_need: float = 0.0

        # Thresholds for drive increase
        self._low_curiosity_threshold = 0.3
        self._active_steps_threshold = 100
        self._social_steps_threshold = 50

    def update(
        self,
        curiosity_reward: float,
        steps_active: int,
        steps_since_social: int,
    ) -> None:
        """Update all drives based on current state.

        Args:
            curiosity_reward: Recent curiosity reward value.
            steps_active: Consecutive steps of active learning.
            steps_since_social: Steps since last caregiver interaction.
        """
        # Curiosity hunger: increases when curiosity reward is low
        if curiosity_reward < self._low_curiosity_threshold:
            self.curiosity_hunger = min(
                1.0, self.curiosity_hunger + 0.01
            )
        else:
            self.curiosity_hunger = max(
                0.0, self.curiosity_hunger - 0.02
            )

        # Rest need: increases with continuous active learning
        rest_ratio = steps_active / max(1, self._active_steps_threshold)
        self.rest_need = min(1.0, rest_ratio * 0.5)

        # Social need: increases with steps since caregiver
        social_ratio = steps_since_social / max(
            1, self._social_steps_threshold
        )
        self.social_need = min(1.0, social_ratio * 0.5)

    def needs_rest(self) -> bool:
        """Check if rest need is critical (> 0.8)."""
        return self.rest_need > 0.8

    def needs_social(self) -> bool:
        """Check if social need is critical (> 0.8)."""
        return self.social_need > 0.8

    def get_drives(self) -> dict[str, float]:
        """Return all drive levels as a dictionary."""
        return {
            "curiosity_hunger": self.curiosity_hunger,
            "rest_need": self.rest_need,
            "social_need": self.social_need,
        }
