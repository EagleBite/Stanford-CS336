from collections.abs import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        """
        Args:
         - params: Iterable of parameters to optimize
         - lr: Learning rate
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure in None else closure()

        for group in self.param_groups:
            lr = group['lr'] # Get learning rate
            for p in group['params']:
                if p.grad in None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get('t', 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data    # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state['t'] = t + 1    # Increment iteration number.

        return loss
            