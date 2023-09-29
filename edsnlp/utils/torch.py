import math
from enum import Enum
from typing import TypeVar

import torch

Args = TypeVar("Args")


def pad_2d(data, pad=0, **kwargs):
    max_len = max(map(len, data), default=0)
    padded = [row + [pad] * (max_len - len(row)) for row in data]
    return torch.as_tensor(padded, **kwargs)


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    glu = "glu"


def get_activation_function(activation: ActivationFunction):
    return getattr(torch.nn.functional, activation)


def mask_to_triangle(mask):
    """
    Convert a boolean mask to a tensor containing distance to the nearest
    edge of the mask, in each direction.
    For example, if the mask is
    ```
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ]
    ```

    The output will be
    ```
    [
        [0, 1, 2, 1, 0],
        [0, 1, 0, -1, -2]
    ]
    ```

    Parameters
    ----------
    mask: torch.Tensor
        A boolean mask

    Returns
    -------
    torch.Tensor
    """
    ramp = torch.arange(0, mask.shape[1], 1)
    scores = torch.min(ramp, mask.sum(1, keepdim=True) - 1 - ramp).view(-1)
    return scores


def make_windows(lengths, size, stride):
    max_len = max(lengths)
    windows = pad_2d(
        [
            list(
                range(
                    idx * stride + doc_idx * max_len,
                    min(idx * stride + size, length) + doc_idx * max_len,
                )
            )
            for doc_idx, length in enumerate(lengths)
            for idx in range(0, 1 + max(0, math.ceil((length - size) / stride)))
        ],
        pad=-1,
    )
    windows_mask = windows != -1
    windows[~windows_mask] = 0
    indexer = torch.zeros((len(lengths), max_len), dtype=torch.long).view(-1)
    scores = mask_to_triangle(windows_mask)
    scores = scores * len(scores) + torch.arange(len(scores))
    scores[~windows_mask.view(-1)] = -1
    indexer.index_reduce_(
        dim=0,
        source=scores,
        index=windows.view(-1),
        reduce="amax",
    )
    indexer %= len(scores)
    return windows.masked_fill(~windows_mask, -1), indexer
