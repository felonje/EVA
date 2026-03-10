"""Presence Dynamics — models engagement/withdrawal/repair cycle.

The caregiver's engagement level naturally varies based on EVA's
behavior quality. Good behavior slowly increases engagement.
Poor behavior quickly decreases it. This models real caregiver
dynamics and teaches EVA about relationship maintenance.
"""

from __future__ import annotations


class PresenceDynamics:
    """Models the caregiver engagement/withdrawal/repair cycle.

    Engagement starts at 1.0 (fully engaged) and moves based on
    EVA's behavior quality. Increase is slow (+0.01), decrease
    is fast (-0.05) — matching real attachment dynamics.
    """

    def __init__(self) -> None:
        self.engagement_level: float = 1.0
        self._history: list[float] = []

    def update(self, behavior_quality: float) -> None:
        """Update engagement based on EVA's behavior quality.

        Args:
            behavior_quality: Quality of EVA's recent behavior [-1, 1].
        """
        self._history.append(behavior_quality)

        if behavior_quality > 0.5:
            # Good behavior: slow increase
            self.engagement_level = min(
                1.0, self.engagement_level + 0.01
            )
        elif behavior_quality < -0.5:
            # Poor behavior: faster decrease
            self.engagement_level = max(
                0.0, self.engagement_level - 0.05
            )
        # Otherwise: no change (neutral behavior)

    def get_response_probability(self) -> float:
        """Probability that caregiver responds to EVA's action.

        Returns:
            Probability [0, 1] based on current engagement level.
        """
        return self.engagement_level

    def is_withdrawn(self) -> bool:
        """Check if caregiver has withdrawn (engagement < 0.3)."""
        return self.engagement_level < 0.3

    def repair(self) -> None:
        """Attempt to repair engagement after withdrawal.

        Raises engagement to 0.5 if currently below — models
        the repair process in attachment relationships.
        """
        if self.engagement_level < 0.5:
            self.engagement_level = 0.5

    def get_history(self) -> list[float]:
        """Return behavior quality history."""
        return list(self._history)
