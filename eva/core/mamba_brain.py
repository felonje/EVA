"""Mamba-based BabyBrain implementation.
A minimal, memory-efficient State Space Model (SSM) for EVA.
Reference: https://github.com/johnma2006/mamba-minimal
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class MambaConfig:
    d_model: int = 256
    n_layers: int = 8
    vocab_size: int = 512
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand
        
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

    def forward(self, x: torch.Tensor):
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

    def ssm(self, x: torch.Tensor):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.config.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        
        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

def einsum(signature, *operands):
    return torch.einsum(signature, *operands)

class MambaBabyBrain(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm_f(x)
        return self.lm_head(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_snapshot(self, sample_ratio: float = 1.0, param_names: list[str] | None = None) -> dict[str, dict[str, float]]:
        snapshot = {}
        param_dict = dict(self.named_parameters())
        targets = param_names if param_names else [n for n, p in param_dict.items() if p.requires_grad]
        
        for name in targets:
            if name in param_dict:
                p = param_dict[name].detach().float()
                snapshot[name] = {"mean": p.mean().item(), "std": p.std().item()}
        return snapshot
