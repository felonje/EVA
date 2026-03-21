"""Test script for Mamba architecture and SFT loop."""
import torch
import logging
from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.environment.nursery import NurseryEnvironment
from eva.training.loop import TrainingLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mamba_sft_integration():
    logger.info("Starting Mamba and SFT integration test...")
    
    # 1. Setup Config Data
    config_data = {
        "model": {
            "d_model": 128,
            "n_layers": 2,
            "vocab_size": 512,
            "random_init": True,
            "dtype": "float32"
        },
        "training": {
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "lr_scheduler": "none"
        },
        "curiosity": {
            "alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2
        },
        "memory": {"max_size": 1000},
        "guidance": {"ai_scaffold": {"response_contingency": 0.8, "socratic_probability": 0.6}},
        "emotions": {"enabled": True},
        "developmental_emotions": {}
    }
    config = EVAConfig(config_data)
    
    # 2. Initialize Components
    tokenizer = EVATokenizer()
    brain = BabyBrain(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        dtype_str="float32",
        device=torch.device("cpu")
    )
    
    environment = NurseryEnvironment(tokenizer)
    loop = TrainingLoop(brain, config, environment, tokenizer)
    
    # 3. Run a few steps of training
    logger.info("Running 50 steps of training...")
    stats = loop.train(num_steps=50, log_every=10)
    
    logger.info("Training stats: %s", stats)
    assert stats["total_steps"] == 50
    assert stats["avg_loss"] > 0
    
    # 4. Check SFT Interface
    logger.info("Checking SFT interface...")
    assert hasattr(loop, 'sft')
    initial_lr = loop.optimizer.param_groups[0]['lr']
    
    # Simulate a reflection
    loop.sft.adjust_learning_rate(1.1)
    new_lr = loop.optimizer.param_groups[0]['lr']
    
    logger.info("Initial LR: %e, New LR: %e", initial_lr, new_lr)
    assert abs(new_lr - initial_lr * 1.1) < 1e-10
    
    logger.info("Mamba and SFT integration test PASSED!")

if __name__ == "__main__":
    test_mamba_sft_integration()
