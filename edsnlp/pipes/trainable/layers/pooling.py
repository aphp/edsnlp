from typing import Optional

import torch
from typing_extensions import Literal


class Pooling(torch.nn.Module):
    """
    Pools groups of word embeddings into one embedding per group, without padding.
    The groups are given like in `torch.nn.functional.embedding_bag`: a flat tensor of
    word indices and the offset of each group in it. It is used by `eds.span_pooler`
    to pool the words of each span.

    Parameters
    ----------
    input_size : int
        Size of the word embeddings
    mode : Literal["max", "sum", "mean", "attention"]
        How the word embeddings of a group are aggregated:

        - `"max"`, `"sum"`, `"mean"`: element-wise reduction over the words
        - `"attention"`: weighted sum of the words, the weights being a softmax over
          the words of the group of scores given by a learned linear layer
    """

    def __init__(
        self,
        input_size: int,
        mode: Literal["max", "sum", "mean", "attention"] = "mean",
    ):
        super().__init__()
        self.mode = mode
        # No bias since a constant score does not change the softmax
        self.attention: Optional[torch.nn.Linear] = (
            torch.nn.Linear(input_size, 1, bias=False) if mode == "attention" else None
        )

    def forward(
        self,
        embeds: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeds : torch.Tensor
            Word embeddings of shape `(n_words, input_size)`, indexed by `indices`
        indices : torch.Tensor
            Flat indices of the words of every group, one group after the other
        offsets : torch.Tensor
            Start of each group in `indices`, a group can be empty

        Returns
        -------
        torch.Tensor
            One embedding per group, of shape `(n_groups, input_size)`, zeros for
            empty groups
        """
        if self.mode != "attention":
            return torch.nn.functional.embedding_bag(
                input=indices,
                weight=embeds,
                offsets=offsets,
                mode=self.mode,
            )

        n_groups = len(offsets)
        lengths = torch.diff(offsets, append=offsets.new_tensor([len(indices)]))
        group_idx = torch.repeat_interleave(
            torch.arange(n_groups, device=offsets.device),
            lengths,
            output_size=len(indices),
        )
        words = embeds[indices]
        scores = self.attention(words).squeeze(-1)
        # Softmax over the words of each group, shifted by the max score of the group
        # to avoid overflows
        max_scores = scores.new_full((n_groups,), float("-inf")).scatter_reduce(
            0, group_idx, scores.detach(), reduce="amax"
        )
        exp_scores = torch.exp(scores - max_scores[group_idx])
        denom = exp_scores.new_zeros(n_groups).index_add(0, group_idx, exp_scores)
        weights = exp_scores / denom[group_idx]
        return words.new_zeros(n_groups, words.size(-1)).index_add(
            0, group_idx, words * weights.unsqueeze(-1)
        )
