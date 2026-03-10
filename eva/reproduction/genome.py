"""Genome — heritable hyperparameters with mutation.

The genome encodes the heritable aspects of an EVA: architecture
parameters, curiosity weights, emotional baselines, etc. During
reproduction, the genome is inherited with small mutations.
Weights are NOT inherited — Ron Protocol demands fresh initialization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Optional


class Genome:
    """Heritable hyperparameters for an EVA.

    The genome contains parameters that are passed from parent to child
    with small mutations. It does NOT contain learned weights — children
    always start with fresh random weights (Ron Protocol).

    Args:
        genes: Dictionary of gene name -> value.
        generation: Generation number.
        parent_genome_hash: Hash of parent's genome (None for gen 1).
    """

    # Default genes for a first-generation EVA
    DEFAULT_GENES: dict[str, Any] = {
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "curiosity_alpha": 0.3,
        "curiosity_beta": 0.3,
        "curiosity_gamma": 0.2,
        "curiosity_delta": 0.2,
        "emotional_ema_rate": 0.1,
        "memory_capacity": 10000,
        "exploration_baseline": 1.0,
        "social_baseline": 0.5,
    }

    # Mutation rates per gene
    MUTATION_RATES: dict[str, float] = {
        "d_model": 0.0,         # Architecture is fixed within a lineage
        "n_layers": 0.0,
        "n_heads": 0.0,
        "curiosity_alpha": 0.05,
        "curiosity_beta": 0.05,
        "curiosity_gamma": 0.05,
        "curiosity_delta": 0.05,
        "emotional_ema_rate": 0.02,
        "memory_capacity": 0.0,
        "exploration_baseline": 0.1,
        "social_baseline": 0.1,
    }

    def __init__(
        self,
        genes: Optional[dict[str, Any]] = None,
        generation: int = 1,
        parent_genome_hash: Optional[str] = None,
    ) -> None:
        self.genes = genes if genes is not None else copy.deepcopy(self.DEFAULT_GENES)
        self.generation = generation
        self.parent_genome_hash = parent_genome_hash

    def mutate(self) -> Genome:
        """Create a mutated copy of this genome for a child.

        Returns:
            New Genome with small random mutations applied.
        """
        child_genes = copy.deepcopy(self.genes)

        for gene_name, value in child_genes.items():
            rate = self.MUTATION_RATES.get(gene_name, 0.0)
            if rate > 0 and isinstance(value, (int, float)):
                mutation = random.gauss(0, rate)
                child_genes[gene_name] = value + mutation

        # Normalize curiosity weights to sum to 1.0
        curiosity_keys = ["curiosity_alpha", "curiosity_beta", "curiosity_gamma", "curiosity_delta"]
        total = sum(max(0.01, child_genes[k]) for k in curiosity_keys)
        for k in curiosity_keys:
            child_genes[k] = max(0.01, child_genes[k]) / total

        return Genome(
            genes=child_genes,
            generation=self.generation + 1,
            parent_genome_hash=self.hash(),
        )

    def hash(self) -> str:
        """Compute a hash of this genome for identification."""
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha256(gene_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "genes": self.genes.copy(),
            "generation": self.generation,
            "parent_genome_hash": self.parent_genome_hash,
            "hash": self.hash(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        return cls(
            genes=data.get("genes", {}),
            generation=data.get("generation", 1),
            parent_genome_hash=data.get("parent_genome_hash"),
        )
