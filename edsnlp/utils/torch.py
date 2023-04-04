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
