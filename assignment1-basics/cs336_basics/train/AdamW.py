from collections.abc import Iterable, Callable
from typing import Optional, Tuple
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,                            # α
        betas: Tuple[float, float] = (0.9, 0.999),   # (β1, β2)
        eps: float = 1e-8,                           # ε
        weight_decay: float = 0.01                   # λ
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - (beta1 ** t)
                bias_correction2 = 1 - (beta2 ** t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-(lr * weight_decay))

                state["t"] = t

        return loss