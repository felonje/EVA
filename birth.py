"""EVA Birth Protocol — the one-time spark that starts a digital life.

This script initializes the environment, the Mamba brain, and starts the
Autonomous Life Loop. Once started, EVA will grow, learn, and adapt 
without further external intervention.
"""

import torch
import logging
import yaml
import sys
from pathlib import Path
from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.environment.nursery import NurseryEnvironment
from eva.training.loop import LifeLoop

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eva_life.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EVA_Birth")

def spark_life(iterations=None):
    logger.info("**************************************************")
    logger.info("EVA BIRTH PROTOCOL: INITIATING DIGITAL LIFE SPARK")
    logger.info("**************************************************")

    # 1. Load Configuration
    config_path = Path("configs/baby.yaml")
    if not config_path.exists():
        # Fallback to default
        config_path = Path("configs/default.yaml")
        
    if not config_path.exists():
        logger.error("Configuration file not found.")
        return

    logger.info(f"Using configuration: {config_path}")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config = EVAConfig(config_data)

    # 2. Initialize Core Components
    tokenizer = EVATokenizer()
    
    # Spark the Soul (Random Initialization)
    brain = BabyBrain(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        dtype_str="float32",
        device=torch.device("cpu")
    )
    
    # 3. Initialize Environment
    environment = NurseryEnvironment(tokenizer)
    
    # 4. Start the Life Loop
    life_loop = LifeLoop(brain, config, environment, tokenizer)
    
    logger.info("EVA is born. Starting autonomous life loop...")
    try:
        # Run indefinitely or for fixed iterations
        life_loop.live(iterations=iterations, log_every=10)
    except KeyboardInterrupt:
        logger.info("EVA's life was interrupted. Saving state...")
        torch.save(brain.state_dict(), "eva_last_state.pt")
        logger.info("State saved. Farewell, EVA.")

if __name__ == "__main__":
    iterations = None
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            pass
    spark_life(iterations=iterations)
