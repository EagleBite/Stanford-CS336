import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = float(eps)

        # g: (d_model,) scaling parameter
        self.g = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        in_type = x.dtype
        x_fp32 = x.to(torch.float32)  # for numerical stability

        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)  # (..., 1)
        result = x_fp32 / rms  # (..., d_model)
        result = result * self.g.to(torch.float32)  # (..., d_model)

        return result.to(in_type)  # (..., d_model)