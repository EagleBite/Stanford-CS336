import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        d_k: int —— dimension of query and key vectors
        max_seq_len: int —— Maximum sequence length that will be inputted
        device: torch.device | None = None —— Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for Rotary Positional Embedding."

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq: (d_k/2,)
        half = d_k // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32).float() * 2.0 / d_k))
        
        # positions: (max_seq_len,)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)

        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (max_seq_len, d_k/2)

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_k)
        # token_positions: (batch_size, seq_len)
        *batch_dims, seq_len, d_k = x.shape

        cos = self.cos[token_positions]  # (batch_size, seq_len, d_k/2)
        sin = self.sin[token_positions]  # (batch_size, seq_len, d_k/2)

        half = d_k // 2
        x_ = x.view(*batch_dims, seq_len, half, 2)  # (batch_size, seq_len, d_k/2, 2)
        x0 = x_[..., 0]  # (batch_size, seq_len, d_k/2)
        x1 = x_[..., 1]  # (batch_size, seq_len, d_k/2)

        y0 = x0 * cos - x1 * sin  # (batch_size, seq_len, d_k/2)
        y1 = x0 * sin + x1 * cos  # (batch_size, seq_len, d_k/2)

        y = torch.stack((y0, y1), dim=-1).view(*batch_dims, seq_len, d_k)  # (batch_size, seq_len, d_k)
        return y