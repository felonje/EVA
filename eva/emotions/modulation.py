"""Emotional Modulation — emotions influence learning and behavior.

Takes affect + homeostasis state and produces modulation signals
that adjust learning rate, memory importance, exploration temperature,
and self-modification risk tolerance.
"""

from __future__ import annotations

from eva.emotions.affect import AffectiveState
from eva.emotions.homeostasis import Homeostasis


class EmotionalModulation:
    """Produces modulation signals from affect and homeostasis.

    Emotions don't just feel — they change how EVA learns, what it
    remembers, and how much it explores. This is biologically inspired:
    stress hormones affect memory consolidation, arousal affects attention.
    """

    def __init__(
        self,
        lr_scale: tuple[float, float] = (0.5, 2.0),
        memory_scale: tuple[float, float] = (0.1, 1.0),
        exploration_scale: tuple[float, float] = (0.3, 2.0),
    ) -> None:
        self._lr_min, self._lr_max = lr_scale
        self._mem_min, self._mem_max = memory_scale
        self._exp_min, self._exp_max = exploration_scale

    def get_learning_rate_multiplier(
        self, affect: AffectiveState, homeostasis: Homeostasis
    ) -> float:
        """Compute learning rate multiplier from emotional state.

        - High arousal + low valence (threat) -> fast learning (2.0)
        - Low arousal + high valence (relaxed) -> slow learning (0.5)
        - Needs rest -> near-zero (0.05) for consolidation

        Args:
            affect: Current affective state.
            homeostasis: Current homeostatic drives.

        Returns:
            Learning rate multiplier, clamped to configured range.
        """
        if homeostasis.needs_rest():
            return 0.05

        # Threat response: high arousal + low valence
        threat = affect.arousal * max(0.0, -affect.valence)
        # Relaxation: low arousal + high valence
        relax = (1.0 - affect.arousal) * max(0.0, affect.valence)

        # Blend toward high multiplier for threat, low for relaxation
        multiplier = 1.0 + threat * 1.0 - relax * 0.5

        return max(self._lr_min, min(self._lr_max, multiplier))

    def get_memory_importance(self, affect: AffectiveState) -> float:
        """Compute memory importance from emotional state.

        High absolute arousal -> high importance (emotional events
        are remembered more strongly).

        Args:
            affect: Current affective state.

        Returns:
            Memory importance score, clamped to configured range.
        """
        # Arousal drives importance — intense experiences are remembered
        importance = affect.arousal

        # Extreme valence (positive or negative) also increases importance
        valence_intensity = abs(affect.valence)
        importance = (importance + valence_intensity) / 2.0

        return max(self._mem_min, min(self._mem_max, importance))

    def get_exploration_temperature(
        self, affect: AffectiveState, homeostasis: Homeostasis
    ) -> float:
        """Compute exploration temperature from emotional state.

        - Low valence + low dominance (anxiety) -> low temp (safety-seeking)
        - High valence + high novelty_feeling (wonder) -> high temp (explore)

        Args:
            affect: Current affective state.
            homeostasis: Current homeostatic drives.

        Returns:
            Exploration temperature, clamped to configured range.
        """
        # Anxiety: low valence + low dominance -> seek safety
        anxiety = max(0.0, -affect.valence) * (1.0 - affect.dominance)
        # Wonder: high valence + high novelty -> explore
        wonder = max(0.0, affect.valence) * affect.novelty_feeling

        temperature = 1.0 + wonder * 1.0 - anxiety * 0.7

        # Curiosity hunger increases exploration
        if homeostasis.curiosity_hunger > 0.5:
            temperature += 0.3

        return max(self._exp_min, min(self._exp_max, temperature))

    def get_self_modification_risk_tolerance(
        self, affect: AffectiveState
    ) -> float:
        """Compute willingness to modify self (architecture, parameters).

        High dominance + positive valence -> willing to take risks.
        Low dominance + negative valence -> conservative.

        Args:
            affect: Current affective state.

        Returns:
            Risk tolerance in [0, 1].
        """
        confidence = affect.dominance * max(0.0, affect.valence + 0.5)
        return max(0.0, min(1.0, confidence))
