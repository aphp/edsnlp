from typing import Optional

import torch
import torch.nn.functional as F
from typing_extensions import Literal


class Metric(torch.nn.Module):
    """
    Metric layer, used for computing similarities between two sets of vectors. A typical
    use case is to compute the similarity between a set of query vectors (input
    embeddings) and a set of concept vectors (output embeddings).

    Parameters
    ----------
    in_features : int
        Size of the input embeddings
    out_features : int
        Size of the output embeddings
    num_groups : int
        Number of groups for the output embeddings, that can be used to filter out
        certain concepts that are not relevant for a given query (e.g. do not compare
        a drug with concepts for diseases)
    metric : Literal["cosine", "dot"]
        Whether to compute the cosine similarity between the input and output embeddings
        or the dot product.
    rescale: Optional[float]
        Rescale the output cosine similarities by a constant factor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int = 0,
        metric: Literal["cosine", "dot"] = "cosine",
        rescale: Optional[float] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer(
            "groups", torch.zeros(num_groups, out_features, dtype=torch.bool)
        )
        self.rescale: float = (
            rescale if rescale is not None else 20.0 if metric == "cosine" else 1.0
        )
        self.metric = metric
        self.register_parameter(
            "bias",
            torch.nn.Parameter(torch.tensor(-0.65 if metric == "cosine" else 0.0))
            if bias
            else None,
        )
        self.reset_parameters()

        self._last_version = None
        self._normalized_weight = None

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def normalized_weight(self):
        if (
            (self.weight._version, id(self.weight)) == self._last_version
            and not self.training
            and self._normalized_weight is not None
        ):
            return self._normalized_weight
        normalized_weight = self.normalize_embedding(self.weight)
        if not self.training and normalized_weight is not self.weight:
            self._normalized_weight = normalized_weight
            self._last_version = (self.weight._version, id(self.weight))
        return normalized_weight

    def normalize_embedding(self, inputs):
        if self.metric == "cosine":
            inputs = F.normalize(inputs, dim=-1)
        return inputs

    def forward(self, inputs, group_indices=None, **kwargs):
        x = F.linear(
            self.normalize_embedding(inputs),
            self.normalized_weight(),
        )
        if self.bias is not None:
            x += self.bias
        if self.rescale != 1.0:
            x *= self.rescale
        if group_indices is not None and len(self.groups):
            x = x.masked_fill(~self.groups[group_indices], -10000)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, rescale={}, groups={}".format(
            self.in_features,
            self.out_features,
            float(self.rescale or 1.0),
            self.groups.shape[0] if self.groups is not None else None,
        )
