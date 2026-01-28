import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # W: (out_features, in_features) weight matrix
        # Store as W, not W.T, to optimize memory access patterns during computation
        self.W = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype))
        
        # Truncated normal initialization:  N(0, 2/(din+dout)), truncated to [-3σ, 3σ]
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features), W: (out_features, in_features) -> (..., out_features)
        return einsum(x, self.W, "... i, o i -> ... o")
