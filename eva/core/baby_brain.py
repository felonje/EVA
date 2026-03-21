"""BabyBrain — The randomly initialized neural network at EVA's core.

Ron Protocol: NO pretrained weights. Every EVA starts from scratch.
Now using a custom minimal Mamba implementation for 4GB RAM efficiency.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def detect_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MambaBlock(nn.Module):
    """Minimal Mamba block implementation for resource-constrained environments."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A and D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)

        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize A and B
        # deltaA = exp(delta * A)
        # deltaB = delta * B
        # Equation: b l d, d n -> b l d n
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        # Equation: b l d, b l n, b l d -> b l d n
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)

        x = torch.zeros((b, d_in, n), device=u.device, dtype=deltaA.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class BabyBrain(nn.Module):
    """The core neural network of an EVA."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dtype_str: str = "float16",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.dtype = torch.float16 if dtype_str == "float16" else torch.float32
        self.device = device if device is not None else detect_device()
        self.architecture = "mamba-minimal"

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder layers
        self.layers = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(n_layers)]
        )

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Store last hidden state
        self._last_hidden: Optional[torch.Tensor] = None

        # Convert to target dtype and move to device
        self.to(dtype=self.dtype, device=self.device)

        # Log architecture info
        logger.info(
            "BabyBrain initialized: arch=%s, params=%d, "
            "est_memory=%.3f GB, d_model=%d, n_layers=%d, device=%s",
            self.architecture,
            self.parameter_count,
            self._estimate_memory_gb(),
            d_model,
            n_layers,
            self.device,
        )

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _estimate_memory_gb(self) -> float:
        """Estimate memory usage in GB."""
        bytes_per_param = 2 if self.dtype == torch.float16 else 4
        return (self.parameter_count * bytes_per_param) / (1024 ** 3)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = self.embedding(input_ids)

        # Encoder layers
        for layer in self.layers:
            x = layer(x) + x

        # Final norm
        hidden = self.norm(x)
        self._last_hidden = hidden.detach()

        # Output projection
        logits = self.output_proj(hidden)

        return logits, hidden

    def predict_next(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict probability distribution over next token."""
        logits, _ = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        return F.softmax(last_logits.float(), dim=-1)

    def get_hidden_state(self) -> torch.Tensor:
        """Return the last hidden state."""
        if self._last_hidden is not None:
            return self._last_hidden
        return torch.zeros(1, 1, self.d_model, device=self.device, dtype=self.dtype)

    def get_parameter_snapshot(
        self, sample_ratio: float = 1.0, param_names: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Lightweight parameter snapshot for information gain."""
        snapshot: dict[str, dict[str, float]] = {}
        param_dict = dict(self.named_parameters())
        
        if param_names is not None:
            params = [(n, param_dict[n]) for n in param_names if n in param_dict]
        else:
            params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
            if sample_ratio < 1.0:
                import random
                k = max(1, int(len(params) * sample_ratio))
                params = random.sample(params, k)

        for name, param in params:
            with torch.no_grad():
                p = param.float()
                snapshot[name] = {
                    "mean": p.mean().item(),
                    "std": p.std().item(),
                }
        return snapshot
