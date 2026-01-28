import math
import torch

def softmaxF(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Find the max element
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    exp_x = torch.exp(x_shifted)
    denom = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / denom

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask (optional): (n, m) boolean, True = allowed, False = blocked
    """
    *batch_dims, n, d_k = Q.shape
    m = K.shape[-2]
    assert K.shape[-1] == d_k, "K last dim must match Q last dim (d_k)."
    assert V.shape[-2] == m, "V seq_len must match K seq_len."
    d_v = V.shape[-1]

    # compute attention scores: (batch..., n, m) = Q @ K^T / sqrt(d_k)
    in_dtype = Q.dtype
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    scores = torch.einsum("... n d, ... m d -> ... n m", Qf, Kf) / math.sqrt(d_k)

    if mask is not None:
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg)

    attn = softmaxF(scores, dim=-1)

    out = torch.einsum("... n m, ...m v -> ... n v", attn, Vf)
    return out.to(in_dtype)