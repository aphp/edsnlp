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
        self, embeddings: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        # shape: samples words dim
        if 0 in embeddings.shape:
            return embeddings.view((*embeddings.shape[:-1], self.linear.out_features))  # type: ignore
        max_k = max(conv.kernel_size[0] for conv in self.convolutions)
        left_pad = (max_k) // 2
        right_pad = (max_k - 1) // 2
        n_samples, n_words, dim = embeddings.shape
        n_words_with_pad = n_words + left_pad + right_pad

        # shape: samples (left_pad... words ...right_pad) dim
        padded_x = F.pad(embeddings, pad=(0, 0, max_k // 2, (max_k - 1) // 2))
        padded_mask = F.pad(mask, pad=(max_k // 2 + (max_k - 1) // 2, 0), value=True)

        # shape: (un-padded sample words) dim
        flat_x = padded_x[padded_mask]

        # Conv-1d expects sample * dim * words
        flat_x = flat_x.permute(1, 0).unsqueeze(0)

        # Apply the convolutions over the flattened input
        conv_results = []
        for conv_idx, conv in enumerate(self.convolutions):
            k = conv.kernel_size[0]
            conv_x = conv(flat_x)
            offset_left = left_pad - (k // 2)
            offset_right = conv_x.size(2) - (right_pad - ((k - 1) // 2))
            conv_results.append(conv_x[0, :, offset_left:offset_right])
        flat_x = torch.cat(conv_results, dim=0)
        flat_x = flat_x.transpose(1, 0)  # n_words * dim

        # Apply the non-linearities
        flat_x = torch.relu(flat_x)
        flat_x = self.linear(flat_x)

        # Reshape the output to the original shape
        new_dim = flat_x.size(-1)
        x = torch.empty(
            n_samples * n_words_with_pad,
            new_dim,
            device=flat_x.device,
            dtype=flat_x.dtype,
        )
        flat_mask = padded_mask.clone()
        flat_mask[-1, padded_mask[-1].sum() - right_pad :] = False
        flat_mask[0, :left_pad] = False
        flat_mask = flat_mask.view(-1)
        x[flat_mask] = flat_x
        x = x.view(n_samples, n_words_with_pad, new_dim)
        x = x[:, left_pad:-right_pad]

        # Apply the residual connection
        if self.residual is not None:
            x = self.residual(embeddings, x)

        return x.masked_fill_((~mask).unsqueeze(-1), 0)
