"""Nursery Environment — safe learning environment for early development.

The nursery provides simple, structured stimuli for EVA's earliest
learning. It generates random character sequences, simple patterns,
and repetitions that allow EVA to begin developing prediction abilities.
"""

from __future__ import annotations

import random
import string
from typing import Any

from eva.core.tokenizer import EVATokenizer
from eva.environment.base import BaseEnvironment


class NurseryEnvironment(BaseEnvironment):
    """Safe learning environment for early EVA development.

    Generates simple stimuli: random characters, repeating patterns,
    and basic sequences. Complexity increases as EVA progresses through
    developmental phases.

    Args:
        tokenizer: The EVA tokenizer for encoding stimuli.
        difficulty: Starting difficulty level (0.0 = simplest, 1.0 = hardest).
    """

    def __init__(
        self, tokenizer: EVATokenizer, difficulty: float = 0.0
    ) -> None:
        super().__init__(name="nursery")
        self._tokenizer = tokenizer
        self._difficulty = max(0.0, min(1.0, difficulty))
        self._current_sequence: list[int] = []
        self._position: int = 0
        self._patterns: list[str] = self._generate_patterns()

    def _generate_patterns(self) -> list[str]:
        """Generate stimulus patterns based on difficulty."""
        patterns: list[str] = []

        # Simple repetitions (always available)
        for char in "abcde":
            patterns.append(char * 10)

        # Simple alternations
        patterns.extend(["ababababab", "abcabcabc", "aabbaabb"])

        if self._difficulty > 0.3:
            # Longer patterns
            patterns.extend([
                "abcdabcdabcd",
                "aabbccaabbcc",
                "abcdeabcde",
            ])

        if self._difficulty > 0.6:
            # More complex patterns
            patterns.extend([
                "the cat sat on the mat",
                "one two three one two three",
                "hello world hello world",
            ])

        if self._difficulty > 0.8:
            # Sentences with structure
            patterns.extend([
                "if it rains then we stay inside",
                "the big dog chased the small cat",
                "I see you and you see me",
            ])

        return patterns

    def reset(self) -> list[int]:
        """Reset with a new random pattern.

        Returns:
            Initial token sequence (first few tokens of pattern).
        """
        pattern = random.choice(self._patterns)
        self._current_sequence = self._tokenizer.encode(pattern)
        self._position = min(3, len(self._current_sequence) - 1)
        self._step_count = 0
        return self._current_sequence[:self._position]

    def step(self, action: int) -> tuple[int, dict[str, Any]]:
        """Reveal the next actual token.

        Args:
            action: EVA's predicted next token.

        Returns:
            Tuple of (actual_token, info_dict).
        """
        if self._position >= len(self._current_sequence):
            # Pattern exhausted — start a new one
            self.reset()

        actual = self._current_sequence[self._position]
        correct = action == actual
        self._position += 1
        self._step_count += 1

        info = {
            "correct": correct,
            "position": self._position,
            "pattern_length": len(self._current_sequence),
            "difficulty": self._difficulty,
        }

        return actual, info

    def get_current_sequence(self) -> list[int]:
        """Return tokens seen so far."""
        return self._current_sequence[:self._position]

    def increase_difficulty(self, amount: float = 0.1) -> None:
        """Increase environment difficulty.

        Args:
            amount: How much to increase difficulty.
        """
        self._difficulty = min(1.0, self._difficulty + amount)
        self._patterns = self._generate_patterns()

    @property
    def difficulty(self) -> float:
        """Current difficulty level."""
        return self._difficulty
