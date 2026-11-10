"""
The heads of the `eds.doc_classifier` component. A head is configured by a plain
dict of options (see `HeadConfig`), from which the component instantiates a
`DocClassifierHead` module.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from spacy.tokens import Doc
from typing_extensions import Literal, TypedDict

#: Target value of a document that has no gold value for a head. It matches the
#: default `ignore_index` of `nn.CrossEntropyLoss`.
IGNORE_INDEX = -100

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class HeadConfig(TypedDict, total=False):
    labels: Optional[Union[List[Any], str, Path]]
    multi_label: bool
    threshold: float
    class_weights: Optional[Union[Dict[Any, float], str, Path]]
    hidden_size: Optional[int]
    activation_mode: Literal["relu", "gelu", "silu"]
    dropout_rate: float
    layer_norm: bool
    loss_weight: float


def _maybe_load(value: Any) -> Any:
    """
    Read `value` from disk if it is a path, else return it as-is. A `.txt` file is
    read as one item per line, any other file as JSON.
    """
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix == ".txt":
            return [line for line in path.read_text().splitlines() if line.strip()]
        return json.loads(path.read_text())
    return value


class DocClassifierHead(nn.Module):
    """
    A head of `eds.doc_classifier`: an optional hidden block (`Linear -> activation
    -> [LayerNorm] -> Dropout`) followed by a linear classifier. A single-label head
    is trained with a cross-entropy loss and decoded with an `argmax`, a multi-label
    head with a binary cross-entropy loss and decoded by keeping every label whose
    probability is at least `threshold`.

    The classifier is created in `__init__`, with no output when the labels are not
    known yet, and resized in place by `update_labels`, so that an optimizer built
    before the labels are known still trains it.
    """

    def __init__(
        self,
        input_size: int,
        *,
        labels: Optional[Union[List[Any], str, Path]] = None,
        multi_label: bool = False,
        threshold: float = 0.5,
        class_weights: Optional[Union[Dict[Any, float], str, Path]] = None,
        hidden_size: Optional[int] = None,
        activation_mode: Literal["relu", "gelu", "silu"] = "relu",
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.multi_label = multi_label
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.freqs: Optional[Dict[Any, float]] = (
            dict(_maybe_load(class_weights)) if class_weights is not None else None
        )

        if hidden_size:
            self.hidden = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                ACTIVATIONS[activation_mode](),
                *([nn.LayerNorm(hidden_size)] if layer_norm else []),
                nn.Dropout(dropout_rate),
            )
            input_size = hidden_size
        else:
            self.hidden = None
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Initializing zero-element tensors is a no-op"
            )
            self.classifier = nn.Linear(input_size, 0)

        self.labels: List[Any] = []
        self.label2id: Dict[Any, int] = {}
        self.class_weights: Optional[torch.Tensor] = None
        if labels is not None:
            self.update_labels(list(_maybe_load(labels)))

    def update_labels(self, labels: List[Any]) -> None:
        """
        Set the labels of the head and resize its classifier in place, keeping the
        weights of the labels it already had.
        """
        old_label2id = self.label2id
        common = [label for label in labels if label in old_label2id]
        new_index = [labels.index(label) for label in common]
        old_index = [old_label2id[label] for label in common]
        old = self.classifier
        new = nn.Linear(old.in_features, len(labels)).to(old.weight.device)
        new.weight.data[new_index] = old.weight.data[old_index]
        new.bias.data[new_index] = old.bias.data[old_index]
        old.weight.data = new.weight.data
        old.bias.data = new.bias.data
        old.out_features = len(labels)

        self.labels = list(labels)
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.class_weights = self._class_weights()

    def _class_weights(self) -> Optional[torch.Tensor]:
        """Inverse-frequency class weights aligned with the labels."""
        if self.freqs is None:
            return None
        total = sum(self.freqs.values())
        weights = torch.zeros(len(self.labels))
        for label, freq in self.freqs.items():
            if label in self.label2id and freq > 0:
                weights[self.label2id[label]] = total / (len(self.labels) * freq)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pooled embeddings to per-label logits."""
        if self.hidden is not None:
            x = self.hidden(x)
        return self.classifier(x)

    def build_target(self, doc: Doc, attr: str) -> torch.Tensor:
        """
        Build the training target of `doc._.<attr>`: a label index for a
        single-label head, a multi-hot vector for a multi-label one. When the
        document has no gold value, the target is filled with `IGNORE_INDEX`, so
        that documents with and without one can be batched together.
        """
        value = getattr(doc._, attr, None)
        if not self.multi_label:
            if value is None:
                return torch.tensor(IGNORE_INDEX, dtype=torch.long)
            if value not in self.label2id:
                raise ValueError(f"Label '{value}' not in the labels of head '{attr}'.")
            return torch.tensor(self.label2id[value], dtype=torch.long)
        if value is None:
            return torch.full((len(self.labels),), IGNORE_INDEX, dtype=torch.float)
        if isinstance(value, str):
            value = [value]
        target = torch.zeros(len(self.labels), dtype=torch.float)
        for label in value:
            if label in self.label2id:
                target[self.label2id[label]] = 1.0
        return target

    @staticmethod
    def is_supervised(target: torch.Tensor) -> torch.Tensor:
        """Boolean mask of the documents of a batch that have a gold value."""
        return target.reshape(len(target), -1)[:, 0] != IGNORE_INDEX

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss, or binary cross-entropy for a multi-label head."""
        weights = (
            self.class_weights.to(logits) if self.class_weights is not None else None
        )
        target = target.to(logits.device)
        if self.multi_label:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=weights
            )
        return nn.functional.cross_entropy(logits, target, weight=weights)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax over the labels, or an independent sigmoid per label."""
        if self.multi_label:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def decode(self, logits: torch.Tensor) -> List[Any]:
        """
        Decode a batch of logits into one prediction per document: a label for a
        single-label head, the list of labels whose probability is at least
        `threshold` for a multi-label one.
        """
        if not self.multi_label:
            return [self.labels[i] for i in logits.argmax(-1).tolist()]
        keep = self.probs(logits) >= self.threshold
        return [
            [self.labels[j] for j in row.nonzero(as_tuple=True)[0].tolist()]
            for row in keep
        ]
