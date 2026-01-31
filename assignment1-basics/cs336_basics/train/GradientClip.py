import torch
from typing import Iterable

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6

    grads = []
    for p in params:
        if p.grad is None:
            continue
        grads.append(p.grad)

    if len(grads) == 0:
        return

    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total_sq = total_sq + (g.detach().float() ** 2).sum()
    total_norm = torch.sqrt(total_sq)

    if total_norm > max_norm:
        scale = float(max_norm) / (total_norm.item() + eps)
        for g in grads:
            g.mul_(scale)  # in-place
