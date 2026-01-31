import torch

def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # subtract max for numerical stability
    m = logits.max(dim=-1, keepdim=True).values
    z = logits - m

    sum_exp = torch.exp(z).sum(dim=-1)
    log_denom = torch.log(sum_exp)

    tgt = z.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = (log_denom - tgt).mean()
    return loss