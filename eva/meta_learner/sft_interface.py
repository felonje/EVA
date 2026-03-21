"""Self-Fine-Tuning (SFT) Interface for EVA.
Allows EVA to introspect its state and perform self-modifications.
"""
from __future__ import annotations
import torch
import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

class SelfAware(Protocol):
    """Protocol for components that can be introspected by EVA."""
    def to_dict(self) -> Dict[str, Any]: ...

class SFTInterface:
    """The bridge between EVA's cognitive process and its training parameters."""
    def __init__(self, optimizer: torch.optim.Optimizer, config: Any):
        self.optimizer = optimizer
        self.config = config
        self._history: List[Dict[str, Any]] = []

    def get_internal_state(self, affect: SelfAware, homeostasis: SelfAware, curiosity: Any) -> Dict[str, Any]:
        """Collects all sensory data available to EVA for self-reflection."""
        state = {
            "affect": affect.to_dict(),
            "homeostasis": homeostasis.to_dict(),
            "curiosity_stats": curiosity.get_stats() if hasattr(curiosity, 'get_stats') else {},
            "current_lrs": [pg['lr'] for pg in self.optimizer.param_groups]
        }
        return state

    def adjust_learning_rate(self, multiplier: float):
        """Allows EVA to directly influence its learning rate."""
        if not (0.1 <= multiplier <= 10.0):
            logger.warning("EVA requested an extreme LR multiplier: %.2f. Clamping to [0.1, 10.0]", multiplier)
            multiplier = max(0.1, min(10.0, multiplier))
            
        for pg in self.optimizer.param_groups:
            pg['lr'] *= multiplier
        logger.info("EVA self-adjusted learning rate by %.2f multiplier", multiplier)

    def update_config(self, key: str, value: Any):
        """Allows EVA to update its own configuration (e.g., curiosity weights)."""
        # Safety check: only allow certain keys
        allowed_keys = ["curiosity.alpha", "curiosity.beta", "curiosity.gamma", "curiosity.delta", "training.max_grad_norm"]
        if key in allowed_keys:
            parts = key.split('.')
            target = self.config
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
            logger.info("EVA self-updated config: %s = %s", key, value)
        else:
            logger.warning("EVA tried to update protected config key: %s", key)

    def log_reflection(self, reflection_text: str):
        """Stores EVA's internal monologue or reasoning about its state."""
        self._history.append({"step": len(self._history), "reflection": reflection_text})
        logger.info("EVA Reflection: %s", reflection_text)
