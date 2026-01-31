import torch
import torch.nn as nn
from .Embedding import Embedding
from .RMSNorm import RMSNorm
from .TransformerBlock import TransformerBlock
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .Linear import Linear
from .Softmax import softmaxF

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        num_layers: int, 
        context_length: int, 
        theta: float = 10000.0, 
        d_ff: int | None = None
    ):
        """
        Args:
         - vocab_size: int —— size of the vocabulary
         - d_model: int —— dimensionality of the model
         - num_heads: int —— number of attention heads
         - num_layers: int —— number of Transformer blocks
         - context_length: int —— maximum sequence length
         - theta: float —— base frequency for Rotary Positional Embedding
         - d_ff: int | None —— dimensionality of the feed-forward inner layer
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = Embedding(vocab_size, d_model)

        rope = RotaryPositionalEmbedding(
            theta=theta,  
            d_k=d_model // num_heads,
            max_seq_len=context_length
        )

        self.layers = nn.ModuleList(
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope)
            for _ in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
    