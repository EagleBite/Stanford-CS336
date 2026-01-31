import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Args:
         - num_embeddings: size of the vocabulary
         - embedding_dim: size of each embedding vector
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize the embedding matrix with truncated normal distribution
        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (Batch_size, Seq_len), each entry is an integer in [0, num_embeddings-1]
        self.W: (num_embeddings, embedding_dim), Lookup table
        
        returns: (Batch_size, Seq_len, embedding_dim)
        """
        return self.W[token_ids]