import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from .Linear import Linear
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .ScaledDotProductAttention import scaled_dot_product_attention

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k

        self.W_Q = Linear(d_model, num_heads * self.d_k)
        self.W_K = Linear(d_model, num_heads * self.d_k)
        self.W_V = Linear(d_model, num_heads * self.d_v)
        self.W_O = Linear(num_heads * self.d_v, d_model)

        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_Q(x) # B, S, H*d_k
        K = self.W_K(x) # B, S, H*d_k
        V = self.W_V(x) # B, S, H*d_v

        # reshape to each head
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        Q = rearrange(Q, "b h s d -> (b h) s d")
        K = rearrange(K, "b h s d -> (b h) s d")
        V = rearrange(V, "b h s d -> (b h) s d")

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = ~torch.triu(
            torch.ones((S, S), device=x.device, dtype=torch.bool), 
            diagonal=1
        )

        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        out = rearrange(out, "(b h) s d -> b s (h d)", b=B, h=self.num_heads)
        out = self.W_O(out)

        return out



