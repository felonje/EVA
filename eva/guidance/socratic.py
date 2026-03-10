"""Socratic Module — generates questions instead of answers.

Rather than telling EVA what to do, the Socratic module asks
questions that encourage EVA to think, predict, and reflect.
Question type is selected based on EVA's current emotional state.
"""

from __future__ import annotations

import random

from eva.emotions.affect import AffectiveState


class SocraticModule:
    """Generates developmental questions based on EVA's state.

    Questions are selected by affect dimensions:
    - Distress (low valence, high arousal): self-foresight questions
    - High social + low valence: empathy questions
    - High dominance + high valence: humility questions
    - High novelty: expansion questions
    - Default: epistemic questions
    """

    def generate_question(
        self, eva_output: str, eva_affect: AffectiveState
    ) -> str:
        """Generate a Socratic question based on EVA's state.

        Args:
            eva_output: What EVA recently produced.
            eva_affect: EVA's current affective state.

        Returns:
            A question string designed to promote thinking.
        """
        keywords = self._extract_keywords(eva_output)
        keyword = keywords[0] if keywords else "that"

        # Select question type based on affect
        if eva_affect.valence < -0.5 and eva_affect.arousal > 0.6:
            # Self-foresight: help EVA predict and plan
            return self._foresight_question(keyword)
        elif eva_affect.social > 0.6 and eva_affect.valence < 0:
            # Empathy: encourage perspective-taking
            return self._empathy_question(keyword)
        elif eva_affect.dominance > 0.8 and eva_affect.valence > 0.5:
            # Humility: gentle challenge to certainty
            return self._humility_question(keyword)
        elif eva_affect.novelty_feeling > 0.7:
            # Expansion: encourage broader exploration
            return self._expansion_question(keyword)
        else:
            # Default: epistemic question
            return self._epistemic_question(keyword)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant words from EVA's output."""
        if not text.strip():
            return []
        words = text.strip().split()
        return [w for w in words if len(w) > 2][:3]

    def _foresight_question(self, keyword: str) -> str:
        """Generate a self-foresight question."""
        templates = [
            f"What do you predict happens next with '{keyword}'?",
            f"If '{keyword}' continues, what might you see?",
            f"What would you expect to follow '{keyword}'?",
        ]
        return random.choice(templates)

    def _empathy_question(self, keyword: str) -> str:
        """Generate an empathy-building question."""
        templates = [
            f"How do you imagine they feel about '{keyword}'?",
            f"What might '{keyword}' mean to someone else?",
            f"If another saw '{keyword}', what would they think?",
        ]
        return random.choice(templates)

    def _humility_question(self, keyword: str) -> str:
        """Generate a humility-promoting question."""
        templates = [
            f"What would prove you wrong about '{keyword}'?",
            f"Is there another way to see '{keyword}'?",
            f"What are you not seeing about '{keyword}'?",
        ]
        return random.choice(templates)

    def _expansion_question(self, keyword: str) -> str:
        """Generate an expansion question for novel situations."""
        templates = [
            f"What else might be true about '{keyword}'?",
            f"Where else might you find '{keyword}'?",
            f"What connects '{keyword}' to what you already know?",
        ]
        return random.choice(templates)

    def _epistemic_question(self, keyword: str) -> str:
        """Generate a default epistemic question."""
        templates = [
            f"What would confirm or disconfirm '{keyword}'?",
            f"How would you test whether '{keyword}' is true?",
            f"What evidence do you have for '{keyword}'?",
        ]
        return random.choice(templates)
