from enum import Enum
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from edsnlp.utils.torch import ActivationFunction, get_activation_function


class NormalizationPlacement(str, Enum):
    pre = "pre"
    post = "post"
    none = "none"


class Residual(torch.nn.Module):
    def __init__(self, normalize: NormalizationPlacement = "pre"):
        super().__init__()
        self.normalize = normalize

    def forward(self, before, after):
        return (
            before + F.layer_norm(after, after.shape[1:])
            if self.normalize == "pre"
            else F.layer_norm(before + after, after.shape[1:])
            if self.normalize == "post"
            else before + after
        )


class TextCnn(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
        residual: bool = True,
        normalize: NormalizationPlacement = "pre",
    ):
        """
        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        output_size: Optional[int]
            Size of the output embeddings
            Defaults to the `input_size`
        out_channels: int
            Number of channels
        kernel_sizes: Sequence[int]
            Window size of each kernel
        activation: str
            Activation function to use
        residual: bool
            Whether to use residual connections
        normalize: NormalizationPlacement
            Whether to normalize before or after the residual connection
        """
        super().__init__()

        if out_channels is None:
            out_channels = input_size
        output_size = input_size if output_size is None else output_size

        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv1d(
                in_channels=input_size,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0,
            )
            for kernel_size in kernel_sizes
        )
        self.linear = torch.nn.Linear(
            in_features=out_channels * len(kernel_sizes),
            out_features=output_size,
        )
        self.activation = get_activation_function(activation)
        self.residual = Residual(normalize=normalize) if residual else None

    def forward(
        self, embeddings: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        # sample word dim -> sample dim word
        x = embeddings.masked_fill(~mask.unsqueeze(-1), 0).permute(0, 2, 1)
        x = torch.cat(
            [
                self.activation(
                    # pad by the appropriate amount on both sides of each sentence
                    conv(
                        F.pad(
                            x,
                            pad=[
                                conv.kernel_size[0] // 2,
                                (conv.kernel_size[0] - 1) // 2,
                            ],
                        )
                    )
                    .permute(0, 2, 1)
                    .masked_fill(~mask.unsqueeze(-1), 0)
                )
                for conv in self.convolutions
            ],
            dim=2,
        )
        x = self.linear(x)
        if self.residual is not None:
            x = self.residual(embeddings, x)

        return x.masked_fill(~mask.unsqueeze(-1), 0)
