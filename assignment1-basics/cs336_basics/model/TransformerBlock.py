import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from .SwiGLUFeedForward import SwiGLUFeedForward
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding = None):
        """
        - d_model: int Dimensionality of the Transformer block inputs.
        - num_heads: int Number of heads to use in multi-head self-attention.
        - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = x + self.attn(self.attn_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x

