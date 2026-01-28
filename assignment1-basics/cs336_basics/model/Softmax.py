import torch

def softmaxF(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Find the max element
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    exp_x = torch.exp(x_shifted)
    denom = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / denom