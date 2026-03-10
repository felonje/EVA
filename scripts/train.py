"""Train an EVA from birth.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --steps 5000
    python scripts/train.py --config configs/default.yaml --checkpoint path/to/resume.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.environment.nursery import NurseryEnvironment
from eva.identity.lineage import LineageTracker
from eva.reproduction.birth import BirthProcess
from eva.training.loop import TrainingLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EVA from birth")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/eva",
        help="Path prefix for saving checkpoints",
    )
    args = parser.parse_args()

    # Load config
    logger.info("Loading config from %s", args.config)
    config = EVAConfig.from_yaml(args.config)

    # Create tokenizer
    tokenizer = EVATokenizer()

    # Create or resume EVA
    if args.checkpoint:
        logger.info("Resuming from checkpoint: %s", args.checkpoint)
        brain = BabyBrain(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            dtype_str=config.model.dtype,
        )
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        brain.load_state_dict(checkpoint["brain_state_dict"])
    else:
        logger.info("Creating first-generation EVA...")
        lineage = LineageTracker()
        birth = BirthProcess(config, lineage)
        result = birth.create_first_generation()
        brain = result["brain"]

    # Create environment
    environment = NurseryEnvironment(tokenizer)

    # Create training loop
    loop = TrainingLoop(brain, config, environment, tokenizer)

    if args.checkpoint:
        loop.load_checkpoint(args.checkpoint)

    # Train
    logger.info("Starting training for %d steps...", args.steps)
    stats = loop.train(
        num_steps=args.steps,
        checkpoint_every=getattr(config.training, "checkpoint_every", 1000),
        log_every=getattr(config.training, "log_every", 10),
        checkpoint_path=args.checkpoint_path,
    )

    logger.info("Training complete!")
    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
