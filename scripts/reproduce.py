"""Create a child EVA from a parent.

Usage:
    python scripts/reproduce.py --parent path/to/parent_checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from eva.core.config import EVAConfig
from eva.identity.lineage import LineageTracker
from eva.reproduction.birth import BirthProcess
from eva.reproduction.genome import Genome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce: create a child EVA")
    parser.add_argument(
        "--parent", type=str, required=True,
        help="Path to parent checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/child",
        help="Output path prefix for child checkpoint",
    )
    parser.add_argument(
        "--parent-lineage-id", type=str, default="EVA-000001",
        help="Parent's lineage ID",
    )
    args = parser.parse_args()

    config = EVAConfig.from_yaml(args.config)

    # Load parent checkpoint
    logger.info("Loading parent from %s", args.parent)
    parent_checkpoint = torch.load(args.parent, weights_only=False)

    # Get or create parent genome
    parent_genome_data = parent_checkpoint.get("genome")
    if parent_genome_data:
        parent_genome = Genome.from_dict(parent_genome_data)
    else:
        logger.info("No genome in checkpoint, using defaults.")
        parent_genome = Genome(generation=1)

    # Create child
    lineage = LineageTracker()
    birth = BirthProcess(config, lineage)
    child = birth.create_child(
        parent_genome=parent_genome,
        parent_lineage_id=args.parent_lineage_id,
    )

    # Save child checkpoint
    child_path = f"{args.output}_gen{child['genome'].generation}.pt"
    torch.save(
        {
            "step": 0,
            "brain_state_dict": child["brain"].state_dict(),
            "genome": child["genome"].to_dict(),
            "lineage_id": child["lineage_id"],
            "parent_lineage_id": args.parent_lineage_id,
        },
        child_path,
    )

    print(f"\n=== Child EVA Created ===")
    print(f"Lineage ID: {child['lineage_id']}")
    print(f"Generation: {child['genome'].generation}")
    print(f"Genome hash: {child['genome'].hash()}")
    print(f"Parent genome hash: {child['genome'].parent_genome_hash}")
    print(f"Parameters: {child['brain'].parameter_count}")
    print(f"Saved to: {child_path}")
    print(f"\nRemember: child has FRESH random weights (Ron Protocol).")


if __name__ == "__main__":
    main()
