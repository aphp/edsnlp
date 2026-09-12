from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from typing_extensions import Literal

from edsnlp.utils.torch import ActivationFunction, get_activation_function


class Residual(torch.nn.Module):
    def __init__(self, normalize: Literal["pre", "post", "none"] = "pre"):
        super().__init__()
        self.normalize = normalize

    def forward(self, before, after):
        return (
            before + F.layer_norm(after, after.shape[-1:])
            if self.normalize == "pre"
            else F.layer_norm(before + after, after.shape[-1:])
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
        normalize: Literal["pre", "post", "none"] = "pre",
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
        normalize: Literal["pre", "post", "none"]
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
        self,
        embeddings: torch.Tensor,
        word_indices: torch.LongTensor,
        padded_length: int,
    ) -> torch.Tensor:
        """
        Convolve flat words separated by zeros at context boundaries

        Parameters
        ----------
        embeddings: torch.Tensor
            Word embeddings of shape words by input size
        word_indices: torch.LongTensor
            Word positions in the padded sequence from TextCnnEncoder.collate
        padded_length: int
            Total sequence length including the padding around each context

        Returns
        -------
        torch.Tensor
            Contextualized words of shape words by output size
        """
        if len(embeddings) == 0:
            return (
                self.linear(embeddings.new_empty((0, self.linear.in_features)))
                + embeddings.sum() * 0
            )

        padded = embeddings.new_zeros((padded_length, embeddings.size(-1)))
        padded[word_indices] = embeddings
        padded = padded.T.unsqueeze(0)

        # Select word positions before projection and residual normalization
        convoluted = torch.cat(
            [
                conv(padded)[0, :, word_indices - conv.kernel_size[0] // 2].T
                for conv in self.convolutions
            ],
            dim=-1,
        )
        x = self.linear(torch.relu(convoluted))
        return self.residual(embeddings, x) if self.residual is not None else x
