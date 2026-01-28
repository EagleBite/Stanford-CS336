import torch
import torch.nn as nn
from .EmbeddingLayer import EmbeddingLayer
from .RMSNorm import RMSNorm
from .TransformerBlock import TransformerBlock
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .Linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, context_length: int, theta: float = 10000.0, d_ff: int | None = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = EmbeddingLayer(vocab_size, d_model)

        repo = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=d_model // num_heads, 
            max_seq_len=context_length
        )

        self.layers = nn.ModuleList(
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope)
            for _ in range(num_layers)
        )

        self.final_norm = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
    