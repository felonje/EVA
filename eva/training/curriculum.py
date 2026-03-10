"""Curriculum — developmental phases for EVA training.

EVA develops through phases, each with different learning priorities:
- Prenatal: basic pattern recognition
- Sensorimotor: input-output mapping, prediction
- Cognitive: abstract patterns, longer sequences
- Social: caregiver interaction, source tagging
- Autonomous: self-directed learning, reduced scaffolding
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


PHASES = ["prenatal", "sensorimotor", "cognitive", "social", "autonomous"]


class DevelopmentalCurriculum:
    """Manages EVA's progression through developmental phases.

    Each phase adjusts environment difficulty, learning parameters,
    and caregiver involvement. Progression is based on competence
    metrics, not time.

    Args:
        starting_phase: Initial developmental phase.
    """

    def __init__(self, starting_phase: str = "prenatal") -> None:
        if starting_phase not in PHASES:
            raise ValueError(f"Unknown phase: {starting_phase}. Must be one of {PHASES}")
        self._current_phase = starting_phase
        self._phase_steps: dict[str, int] = {p: 0 for p in PHASES}
        self._competence: dict[str, float] = {
            "prediction": 0.0,
            "pattern_recognition": 0.0,
            "sequence_memory": 0.0,
            "social_interaction": 0.0,
            "self_direction": 0.0,
        }

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def phase_index(self) -> int:
        return PHASES.index(self._current_phase)

    def get_phase_config(self) -> dict[str, Any]:
        """Get learning parameters for the current phase."""
        configs = {
            "prenatal": {
                "difficulty": 0.0,
                "caregiver_involvement": 1.0,
                "exploration_bonus": 0.5,
                "min_steps_to_advance": 500,
                "advancement_threshold": 0.3,
            },
            "sensorimotor": {
                "difficulty": 0.2,
                "caregiver_involvement": 0.9,
                "exploration_bonus": 0.7,
                "min_steps_to_advance": 1000,
                "advancement_threshold": 0.5,
            },
            "cognitive": {
                "difficulty": 0.5,
                "caregiver_involvement": 0.7,
                "exploration_bonus": 1.0,
                "min_steps_to_advance": 2000,
                "advancement_threshold": 0.6,
            },
            "social": {
                "difficulty": 0.7,
                "caregiver_involvement": 0.5,
                "exploration_bonus": 0.8,
                "min_steps_to_advance": 3000,
                "advancement_threshold": 0.7,
            },
            "autonomous": {
                "difficulty": 1.0,
                "caregiver_involvement": 0.2,
                "exploration_bonus": 1.0,
                "min_steps_to_advance": float("inf"),
                "advancement_threshold": 1.0,
            },
        }
        return configs[self._current_phase]

    def update_competence(self, metric: str, value: float) -> None:
        """Update a competence metric.

        Args:
            metric: Name of the competence metric.
            value: New value (EMA blended with existing).
        """
        if metric in self._competence:
            old = self._competence[metric]
            self._competence[metric] = old * 0.95 + value * 0.05

    def step(self) -> bool:
        """Record a step and check for phase advancement.

        Returns:
            True if phase advanced.
        """
        self._phase_steps[self._current_phase] += 1
        return self._check_advancement()

    def _check_advancement(self) -> bool:
        """Check if EVA should advance to the next phase."""
        phase_cfg = self.get_phase_config()
        steps_in_phase = self._phase_steps[self._current_phase]

        if steps_in_phase < phase_cfg["min_steps_to_advance"]:
            return False

        # Check competence threshold
        avg_competence = sum(self._competence.values()) / len(self._competence)
        if avg_competence < phase_cfg["advancement_threshold"]:
            return False

        # Advance
        current_idx = PHASES.index(self._current_phase)
        if current_idx < len(PHASES) - 1:
            old_phase = self._current_phase
            self._current_phase = PHASES[current_idx + 1]
            logger.info(
                "PHASE ADVANCEMENT: %s -> %s (avg_competence=%.3f)",
                old_phase, self._current_phase, avg_competence,
            )
            return True

        return False

    def get_competence(self) -> dict[str, float]:
        return self._competence.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_phase": self._current_phase,
            "phase_steps": self._phase_steps.copy(),
            "competence": self._competence.copy(),
        }
