"""Human Interface — CLI-based human interaction with source tagging.

Provides the interface for human caregivers to interact with EVA.
All human input is tagged with <HUMAN> source token per the Covenant.
"""

from __future__ import annotations

import logging
from typing import Optional

from eva.core.tokenizer import EVATokenizer

logger = logging.getLogger(__name__)


class HumanInterface:
    """CLI-based interface for human-EVA interaction.

    All human messages are tagged with the <HUMAN> source token
    to maintain Covenant honesty. The interface tracks interaction
    history for recency and contingency calculations.
    """

    def __init__(self, tokenizer: EVATokenizer) -> None:
        self._tokenizer = tokenizer
        self._interaction_count: int = 0
        self._last_interaction_step: int = 0
        self._history: list[dict[str, str]] = []

    def get_input(self, prompt: str = "Human> ") -> Optional[str]:
        """Get input from human caregiver via CLI.

        Args:
            prompt: The prompt to display.

        Returns:
            Human's input text, or None if empty/EOF.
        """
        try:
            text = input(prompt).strip()
            if not text:
                return None
            self._interaction_count += 1
            self._history.append({"role": "human", "text": text})
            return text
        except EOFError:
            return None

    def encode_human_message(self, text: str) -> list[int]:
        """Encode human text with proper source tagging.

        Args:
            text: Raw human input text.

        Returns:
            Token IDs with <HUMAN> source tag.
        """
        return self._tokenizer.encode(text, source="human")

    def format_eva_output(self, text: str) -> str:
        """Format EVA's output for display to human.

        Args:
            text: EVA's raw output text.

        Returns:
            Formatted string for CLI display.
        """
        return f"EVA> {text}"

    def record_interaction(self, step: int) -> None:
        """Record that an interaction occurred at this step.

        Args:
            step: Current training/interaction step.
        """
        self._last_interaction_step = step

    def get_recency(self, current_step: int) -> float:
        """Get recency of last interaction as a [0, 1] value.

        Args:
            current_step: Current step number.

        Returns:
            Recency score (1.0 = just happened, decays toward 0).
        """
        if self._interaction_count == 0:
            return 0.0
        steps_since = current_step - self._last_interaction_step
        return 1.0 / (1.0 + steps_since * 0.01)

    @property
    def interaction_count(self) -> int:
        """Total number of human interactions."""
        return self._interaction_count

    @property
    def history(self) -> list[dict[str, str]]:
        """Interaction history."""
        return list(self._history)
