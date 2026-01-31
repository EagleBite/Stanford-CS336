from __future__ import annotations

import numpy as np
import torch

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        - x: Numpy array of shape (num_tokens,) containing token indices.
        - batch_size: Number of sequences to include in the batch.
        - context_length: Length of each sequence.
        - device: Device to place the returned tensors on.

    Returns:
        - inputs: Tensor of shape (batch_size, context_length) containing input token indices.
        - targets: Tensor of shape (batch_size, context_length) containing target token indices.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array of token ids, got shape {x.shape}")

    num_tokens = int(x.shape[0])
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)

    max_start = num_tokens - context_length - 1
    if max_start < 0:
        raise ValueError(f"x is too short (len={num_tokens}) for context_length={context_length}")

    for i in range(batch_size):
        start_idx = np.random.randint(0, max_start + 1)
        inputs[i] = x[start_idx : start_idx + context_length]
        targets[i] = x[start_idx + 1 : start_idx + context_length + 1]

    inputs_tensor = torch.tensor(inputs, device=device)
    targets_tensor = torch.tensor(targets, device=device)

    return inputs_tensor, targets_tensor