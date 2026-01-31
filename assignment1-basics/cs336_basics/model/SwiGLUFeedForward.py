import torch
import torch.nn as nn
from .Linear import Linear

def SiLUF(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLUFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int((8 * self.d_model) / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # round to multiple of 64 for efficiency
        self.d_ff = d_ff

        self.W1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        a = self.W1(x) # (..., d_ff)
        b = self.W3(x) # (..., d_ff)

        # SiLU(a) = a * sigmoid(a)
        silua = a * torch.sigmoid(a)

        # gated = SiLU(W1(x)) * W3(x)
        gated = silua * b

        return self.W2(gated)