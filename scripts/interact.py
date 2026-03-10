"""Interactive CLI for communicating with a trained EVA.

Usage:
    python scripts/interact.py --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.emotions.affect import AffectiveState
from eva.guidance.caregiver import AICaregiver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with EVA")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Maximum tokens to generate per response",
    )
    args = parser.parse_args()

    config = EVAConfig.from_yaml(args.config)
    tokenizer = EVATokenizer()

    # Load brain
    brain = BabyBrain(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dtype_str=config.model.dtype,
    )
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    brain.load_state_dict(checkpoint["brain_state_dict"])
    brain.eval()

    affect = AffectiveState()
    caregiver = AICaregiver()

    print("\n=== EVA Interactive Session ===")
    print("Type your message. Type 'quit' to exit.")
    print("Type 'status' to see EVA's emotional state.\n")

    while True:
        try:
            user_input = input("Human> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "status":
            print(f"  Affect: {affect.to_dict()}")
            continue

        # Encode human input
        input_tokens = tokenizer.encode(user_input, source="human")
        input_ids = torch.tensor([input_tokens], dtype=torch.long)

        # Generate EVA response
        generated: list[int] = list(input_tokens)
        with torch.no_grad():
            for _ in range(args.max_tokens):
                context = torch.tensor(
                    [generated[-512:]], dtype=torch.long
                )
                predicted_dist = brain.predict_next(context)

                # Sample from distribution with temperature
                temperature = 0.8
                scaled = predicted_dist / temperature
                probs = torch.softmax(scaled.float(), dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), 1).item()

                if next_token == 3:  # EOS
                    break
                generated.append(next_token)

        # Decode response
        response_tokens = generated[len(input_tokens):]
        response = tokenizer.decode(response_tokens)

        print(f"EVA> {response}")

        # Caregiver occasionally responds
        caregiver_response = caregiver.respond(response, affect)
        if caregiver_response:
            print(f"  [{caregiver_response.emotional_state}] {caregiver_response.text}")


if __name__ == "__main__":
    main()
