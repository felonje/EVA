"""Life Loop — autonomous, spontaneous existence for EVA.

The core loop where EVA lives, learns, and reflects:
1. Exist: Continuous loop of experience and reflection.
2. Sense: Observe environment or internal memories.
3. Act/Think: Predict next token or reflect on past episodes.
4. Feel: Update affective state including boredom and drive.
5. Regulate: SFT interface for autonomous parameter adjustment.
6. Dream: Spontaneous memory consolidation and rehearsal.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.curiosity.reward import CuriosityEngine
from eva.emotions.affect import AffectiveState
from eva.emotions.developmental import CrisisDetector, DevelopmentalEmotions
from eva.emotions.homeostasis import Homeostasis
from eva.emotions.modulation import EmotionalModulation
from eva.environment.base import BaseEnvironment
from eva.guidance.caregiver import AICaregiver
from eva.guidance.presence import PresenceDynamics
from eva.memory.episodic import Episode, EpisodicMemory
from eva.training.curriculum import DevelopmentalCurriculum
from eva.meta_learner.sft_interface import SFTInterface

logger = logging.getLogger(__name__)


class LifeLoop:
    """The autonomous existence loop for EVA."""

    def __init__(
        self,
        brain: BabyBrain,
        config: EVAConfig,
        environment: BaseEnvironment,
        tokenizer: EVATokenizer,
    ) -> None:
        self.brain = brain
        self.config = config
        self.environment = environment
        self.tokenizer = tokenizer

        # Device
        self.device = brain.device

        # Training hyperparameters
        self._grad_accum_steps = getattr(config.training, "gradient_accumulation_steps", 1)
        self._max_grad_norm = getattr(config.training, "max_grad_norm", 1.0)
        self._info_gain_sample_ratio = getattr(config.training, "info_gain_sample_ratio", 1.0)

        # Systems
        self.curiosity = CuriosityEngine(
            alpha=getattr(config.curiosity, "alpha", 0.3),
            beta=getattr(config.curiosity, "beta", 0.3),
            gamma=getattr(config.curiosity, "gamma", 0.2),
            delta=getattr(config.curiosity, "delta", 0.2),
        )
        self.affect = AffectiveState()
        self.homeostasis = Homeostasis()
        self.modulation = EmotionalModulation()
        self.memory = EpisodicMemory(max_size=getattr(config.memory, "max_size", 10000))
        self.curriculum = DevelopmentalCurriculum(starting_phase=getattr(config.training, "phase", "prenatal"))
        
        # Optimizer & SFT
        self.optimizer = torch.optim.Adam(brain.parameters(), lr=getattr(config.training, "learning_rate", 1e-4))
        self.sft = SFTInterface(self.optimizer, self.config)

        # State tracking
        self._step: int = 0
        self._steps_since_social: int = 0
        self._steps_active: int = 0
        self._recent_outcomes: list[torch.Tensor] = []
        self._is_sleeping: bool = False
        self._sft_lr_multiplier: float = 1.0

    def live(self, iterations: Optional[int] = None, log_every: int = 100):
        """EVA starts its life. Can run for fixed iterations or indefinitely."""
        self.brain.train()
        self.environment.reset()
        
        step_count = 0
        while iterations is None or step_count < iterations:
            self._step += 1
            step_count += 1
            self._steps_since_social += 1

            # 1. Decide: Interact or Reflect?
            # Spontaneous action is driven by intrinsic drive and boredom.
            if self.affect.intrinsic_drive > 0.8 and random.random() < 0.3:
                self._reflect()
            elif self.affect.boredom > 0.7 and random.random() < 0.5:
                self._seek_novelty()
            else:
                self._interact()

            # 2. Self-Regulation (SFT)
            if self._step % 100 == 0:
                self._self_reflect()

            # 3. Sleep/Dream Cycle
            if self.affect.arousal < 0.2 and self.affect.valence > 0.3:
                self._dream()

            if self._step % log_every == 0:
                self._log_status()

    def _interact(self):
        """Standard interaction with the environment."""
        context = self.environment.get_current_sequence()
        if len(context) < 2:
            self.environment.reset()
            return

        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        
        # Curiosity preparation
        self.curiosity.prepare(self.brain, sample_ratio=self._info_gain_sample_ratio)

        # Forward pass
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            predicted_dist = self.brain.predict_next(input_ids)
            hidden = self.brain.get_hidden_state()

        predicted_token = predicted_dist.argmax(dim=-1).item()
        actual_token, env_info = self.environment.step(predicted_token)
        
        # Update weights
        self._update_weights(predicted_dist, actual_token)

        # Compute reward and update affect
        reward, reward_breakdown = self.curiosity.compute_reward(
            predicted_dist.detach(), actual_token, self.brain, hidden, 
            self._recent_outcomes, sample_ratio=self._info_gain_sample_ratio
        )
        
        self.affect.update(
            prediction_success=1.0 if env_info.get("correct") else 0.0,
            prediction_error=reward_breakdown["prediction_error"],
            action_success=1.0 if env_info.get("correct") else 0.0,
            caregiver_recency=1.0 / (1.0 + self._steps_since_social * 0.01),
            caregiver_contingency=0.8,
            novelty_signal=reward_breakdown["novelty"]
        )

        # Update homeostasis drives
        self._steps_active += 1
        self.homeostasis.update(
            curiosity_reward=reward,
            steps_active=self._steps_active,
            steps_since_social=self._steps_since_social,
        )

        # Rest period (memory consolidation)
        if self.homeostasis.needs_rest():
            self.memory.consolidate()
            self._steps_active = 0

        # Store memory
        self._store_memory(hidden, predicted_token, actual_token, reward_breakdown["prediction_error"])

    def _reflect(self):
        """Spontaneous internal reflection (replaying memories)."""
        if self.memory.size() < 10:
            return
            
        # Recall a random important memory
        query = torch.randn(1, self.brain.d_model, device=self.device)
        episodes = self.memory.recall(query, k=1)
        if not episodes:
            return
            
        ep = episodes[0]
        logger.info("EVA is reflecting on a memory from step %d", ep.timestamp)
        
        # Internal rehearsal (pseudo-update)
        # We don't have the full context here, so we just simulate a "thought"
        self.affect.update(
            prediction_success=0.5, # Neutral
            prediction_error=ep.surprise * 0.5,
            action_success=0.5,
            caregiver_recency=1.0 / (1.0 + self._steps_since_social * 0.01),
            caregiver_contingency=0.8,
            novelty_signal=0.1 # Reflection is less novel than new experience
        )

    def _dream(self):
        """Sleep cycle: memory consolidation and weight stabilization."""
        logger.info("EVA is entering a dream state (memory consolidation)...")
        merged = self.memory.consolidate()
        logger.info("EVA consolidated %d memories during sleep.", merged)
        # Reduce arousal after sleep
        self.affect.arousal *= 0.5
        self.affect.boredom *= 0.2

    def _seek_novelty(self):
        """Active search for more complex patterns when bored."""
        logger.info("EVA is feeling bored. Seeking more complex stimuli...")
        if hasattr(self.environment, 'increase_difficulty'):
            self.environment.increase_difficulty(0.05)
        self._interact()

    def _self_reflect(self):
        """Autonomous parameter adjustment via SFT."""
        state = self.sft.get_internal_state(self.affect, self.homeostasis, self.curiosity)
        if state['affect']['valence'] < -0.6:
            self._sft_lr_multiplier *= 0.8
            self.sft.log_reflection("I am stressed. Slowing down to stabilize.")
        elif state['affect']['intrinsic_drive'] > 0.9:
            self._sft_lr_multiplier *= 1.2
            self.sft.log_reflection("I feel highly motivated. Accelerating growth.")
        # Clamp SFT multiplier to safe range
        self._sft_lr_multiplier = max(0.1, min(10.0, self._sft_lr_multiplier))

    def _update_weights(self, predicted_dist, actual_token):
        target = torch.tensor([actual_token], device=self.device)
        log_probs = torch.log(predicted_dist.float().squeeze(0) + 1e-10)
        loss = F.nll_loss(log_probs.unsqueeze(0), target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self._max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Apply emotional modulation and SFT multiplier to LR
        lr_mult = self.modulation.get_learning_rate_multiplier(self.affect, self.homeostasis)
        base_lr = getattr(self.config.training, "learning_rate", 1e-4)
        for pg in self.optimizer.param_groups:
            pg["lr"] = base_lr * lr_mult * self._sft_lr_multiplier

    def _store_memory(self, hidden, action, outcome, surprise):
        importance = self.modulation.get_memory_importance(self.affect)
        state_emb = hidden.mean(dim=1).squeeze(0).detach() if hidden is not None else torch.zeros(self.brain.d_model, device=self.device)
        episode = Episode(
            state_embedding=state_emb,
            action=action,
            outcome=outcome,
            surprise=surprise,
            emotional_importance=importance,
            source_tag="self",
            timestamp=self._step,
        )
        self.memory.store(episode)

    def _log_status(self):
        logger.info(
            "Step %d | Valence: %.2f | Arousal: %.2f | Boredom: %.2f | Drive: %.2f | LR: %.2e",
            self._step, self.affect.valence, self.affect.arousal, 
            self.affect.boredom, self.affect.intrinsic_drive, 
            self.optimizer.param_groups[0]["lr"]
        )
