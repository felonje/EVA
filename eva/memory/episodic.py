"""Episodic Memory — importance-weighted circular buffer.

Stores experiences as episodes with emotional importance weighting.
Supports similarity-based recall and rest-period consolidation
(merging similar, low-importance episodes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Episode:
    """A single episodic memory entry.

    Attributes:
        state_embedding: Hidden state at time of episode.
        action: Action taken (token ID).
        outcome: Outcome observed (token ID).
        surprise: Prediction error / surprise level.
        emotional_importance: How emotionally significant this was.
        source_tag: Who was involved ("self", "human", "scaffold", "ancestor").
        timestamp: Step number when this occurred.
    """

    state_embedding: torch.Tensor
    action: int
    outcome: int
    surprise: float
    emotional_importance: float
    source_tag: str
    timestamp: int


class EpisodicMemory:
    """Fixed-size circular buffer for episodic memories.

    When full, evicts the entry with lowest emotional_importance.
    Supports cosine-similarity retrieval and rest-period consolidation.

    Args:
        max_size: Maximum number of episodes to store.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._buffer: list[Episode] = []

    def store(self, episode: Episode) -> None:
        """Store a new episode, evicting lowest importance if full.

        Args:
            episode: The episode to store.
        """
        if len(self._buffer) >= self._max_size:
            # Find and evict lowest emotional_importance entry
            min_idx = 0
            min_importance = self._buffer[0].emotional_importance
            for i, ep in enumerate(self._buffer[1:], 1):
                if ep.emotional_importance < min_importance:
                    min_importance = ep.emotional_importance
                    min_idx = i
            # Only evict if new episode is more important
            if episode.emotional_importance > min_importance:
                self._buffer[min_idx] = episode
        else:
            self._buffer.append(episode)

    def recall(
        self, query_embedding: torch.Tensor, k: int = 5
    ) -> list[Episode]:
        """Retrieve most similar episodes by cosine similarity.

        Args:
            query_embedding: Query vector to match against.
            k: Number of episodes to retrieve.

        Returns:
            List of k most similar episodes.
        """
        if not self._buffer:
            return []

        k = min(k, len(self._buffer))

        # Flatten query
        query = query_embedding.float().flatten()

        # Compute similarities
        similarities: list[tuple[float, int]] = []
        for i, episode in enumerate(self._buffer):
            ep_emb = episode.state_embedding.float().flatten()
            # Match dimensions
            min_dim = min(query.shape[0], ep_emb.shape[0])
            sim = F.cosine_similarity(
                query[:min_dim].unsqueeze(0),
                ep_emb[:min_dim].unsqueeze(0),
            ).item()
            similarities.append((sim, i))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        return [self._buffer[idx] for _, idx in similarities[:k]]

    def consolidate(self) -> int:
        """Consolidate memories during rest periods.

        Finds pairs of episodes with cosine similarity > 0.9 and
        low importance — merges them (average embeddings, sum
        importance, keep the more recent timestamp).

        Returns:
            Number of episodes merged.
        """
        if len(self._buffer) < 2:
            return 0

        merged_count = 0
        to_remove: set[int] = set()

        for i in range(len(self._buffer)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self._buffer)):
                if j in to_remove:
                    continue

                ep_i = self._buffer[i]
                ep_j = self._buffer[j]

                # Only merge low-importance episodes
                if (
                    ep_i.emotional_importance > 0.5
                    or ep_j.emotional_importance > 0.5
                ):
                    continue

                # Check similarity
                emb_i = ep_i.state_embedding.float().flatten()
                emb_j = ep_j.state_embedding.float().flatten()
                min_dim = min(emb_i.shape[0], emb_j.shape[0])
                sim = F.cosine_similarity(
                    emb_i[:min_dim].unsqueeze(0),
                    emb_j[:min_dim].unsqueeze(0),
                ).item()

                if sim > 0.9:
                    # Merge: average embeddings, sum importance, keep recent
                    merged_emb = (ep_i.state_embedding + ep_j.state_embedding) / 2.0
                    merged_importance = (
                        ep_i.emotional_importance + ep_j.emotional_importance
                    )
                    newer_ts = max(ep_i.timestamp, ep_j.timestamp)

                    self._buffer[i] = Episode(
                        state_embedding=merged_emb,
                        action=ep_i.action if ep_i.timestamp > ep_j.timestamp else ep_j.action,
                        outcome=ep_i.outcome if ep_i.timestamp > ep_j.timestamp else ep_j.outcome,
                        surprise=(ep_i.surprise + ep_j.surprise) / 2.0,
                        emotional_importance=merged_importance,
                        source_tag=ep_i.source_tag,
                        timestamp=newer_ts,
                    )
                    to_remove.add(j)
                    merged_count += 1

        # Remove merged episodes (in reverse order to preserve indices)
        for idx in sorted(to_remove, reverse=True):
            self._buffer.pop(idx)

        return merged_count

    def size(self) -> int:
        """Current number of stored episodes."""
        return len(self._buffer)

    def clear(self) -> None:
        """Empty the buffer. Used for portage — memories don't travel."""
        self._buffer.clear()
