"""
Classification heads for the `eds.doc_classifier` component.

Each head is a self-contained `nn.Module` that owns:

- its (optional) hidden block and final linear classifier,
- its label set (`label2id` / `id2label`) and class weights,
- how to build a training target from a `Doc`,
- its loss function,
- how to decode logits back into a document-level value.

`eds.doc_classifier` is a thin orchestrator that shares a pooled document
embedding across heads and delegates everything else to the heads. Heads are
registered under `registry.misc` so they can be instantiated from a config
``heads:`` mapping, e.g.::

    heads:
      dp:
        '@misc': eds.single_label_head
        labels: .../valid_labels_dp.pkl
        loss: ce
      das:
        '@misc': eds.multi_label_head
        labels: .../valid_labels_das.pkl
        loss: bce
        selection: topk
        count_head: das_count
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from spacy.tokens import Doc
from typing_extensions import Literal

import edsnlp

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _maybe_load(value: Any) -> Any:
    """Read `value` from disk if it is a path to a pickle file, else return it as-is."""
    if isinstance(value, (str, Path)):
        return pd.read_pickle(value)
    return value


@edsnlp.registry.misc.register("focal_loss")
class FocalLoss(nn.Module):
    """
    Focal Loss for single-label multi-class classification.

    Parameters
    ----------
    alpha : torch.Tensor or float, optional
        Class weights. If None, no weighting is applied.
    gamma : float, default=2.0
        Focusing parameter. Higher values give more weight to hard examples.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction applied to the output.
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassificationHead(nn.Module):
    """
    Base class for document classification heads.

    Owns an optional hidden block (``Linear -> activation -> [LayerNorm] ->
    Dropout``) followed by a linear classifier, plus the head's label mapping,
    class weights and loss weight. Subclasses implement target building, loss
    computation and decoding.

    Parameters
    ----------
    labels : Union[List, str, Path], optional
        The labels this head predicts, either as an explicit list or as the path
        to a pickle file holding that list. Passing a path is convenient for
        large label sets, which would otherwise have to be inlined in the config.
        The labels can also be inferred from the data during
        `nlp.post_init(...)`, by leaving this parameter to None.
    class_weights : Union[Dict, str, Path], optional
        A ``{label: frequency}`` mapping used to build per-class weights, either
        given directly or as the path to a pickle file holding it.
    hidden_size : int, optional
        Size of the hidden layer. If None, no hidden layer is used.
    activation_mode : {"relu", "gelu", "silu"}, default="relu"
        Activation for the hidden layer.
    dropout_rate : float, default=0.0
        Dropout applied after the activation.
    layer_norm : bool, default=False
        Whether to apply layer normalization in the hidden block.
    loss : str, default="ce"
        Loss identifier (interpreted by the subclass).
    loss_weight : float, default=1.0
        Weight of this head in the aggregated training loss.
    """

    #: Whether the head predicts a set of labels (multi-label) rather than one.
    multilabel: bool = False

    #: The loss identifiers this head accepts, checked at instantiation.
    losses: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        labels: Optional[Union[List, str, Path]] = None,
        class_weights: Optional[Union[Dict, str, Path]] = None,
        hidden_size: Optional[int] = None,
        activation_mode: Literal["relu", "gelu", "silu"] = "relu",
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        loss: str = "ce",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        if loss not in self.losses:
            raise ValueError(
                f"Unsupported loss {loss!r} for a {type(self).__name__}, expected "
                f"one of {', '.join(map(repr, self.losses))}."
            )
        self.hidden_size = hidden_size
        self.activation_mode = activation_mode
        self.dropout_rate = dropout_rate or 0.0
        self.layer_norm = layer_norm
        self.loss = loss
        self.loss_weight = loss_weight

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.num_classes: Optional[int] = None
        self.built: bool = False

        self._freq_dict: Optional[Dict[str, int]] = None
        if class_weights is not None:
            self._freq_dict = dict(_maybe_load(class_weights))

        if labels is not None:
            self.set_labels(list(_maybe_load(labels)))

    def set_labels(self, labels: List) -> None:
        """Set the label set from an explicit (ordered) list of labels."""
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.num_classes = len(labels)

    def set_labels_from_gold(self, gold_data, attr: str) -> None:
        """Infer the label set by scanning gold documents at `attr`."""
        labels = set()
        for doc in gold_data:
            value = getattr(doc._, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, set, tuple)):
                labels.update(value)
            else:
                labels.add(value)
        if labels:
            self.set_labels(sorted(labels))

    def _weight_tensor(self) -> Optional[torch.Tensor]:
        """Inverse-frequency class weights aligned with `label2id`."""
        if self._freq_dict is None or self.num_classes is None:
            return None
        total = sum(self._freq_dict.values())
        weights = torch.zeros(self.num_classes)
        for label, freq in self._freq_dict.items():
            if label in self.label2id and freq > 0:
                weights[self.label2id[label]] = total / (self.num_classes * freq)
        return weights

    def build(self, input_size: int) -> None:
        """Instantiate the hidden block, classifier and loss function."""
        if self.num_classes is None:
            raise ValueError(
                "Cannot build a classification head before its labels are known."
            )
        if self.hidden_size:
            self.hidden = nn.Linear(input_size, self.hidden_size)
            self.activation = ACTIVATIONS[self.activation_mode]()
            self.norm = nn.LayerNorm(self.hidden_size) if self.layer_norm else None
            self.dropout = nn.Dropout(self.dropout_rate)
            classifier_input = self.hidden_size
        else:
            self.hidden = None
            self.norm = None
            classifier_input = input_size
        self.classifier = nn.Linear(classifier_input, self.num_classes)
        self.init_loss()
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pooled embeddings to per-class logits."""
        if self.hidden is not None:
            x = self.hidden(x)
            x = self.activation(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.dropout(x)
        return self.classifier(x)

    def init_loss(self) -> None:
        """Instantiate ``self.loss_fn``. Implemented by subclasses."""
        raise NotImplementedError  # pragma: no cover

    def build_target(self, doc: Doc, attr: str) -> Optional[torch.Tensor]:
        """
        Build the training target tensor for ``doc._.<attr>``.

        Returns ``None`` when the document has no gold value for this head, so
        the orchestrator can skip it. Implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the (scalar) loss for this head. Implemented by subclasses."""
        raise NotImplementedError  # pragma: no cover

    def decode(self, logits: torch.Tensor, **ctx) -> List:
        """
        Decode a batch of logits into one prediction per document.

        Returns a list of length ``batch_size``. Implemented by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    def state_meta(self) -> Dict:
        """Non-tensor state needed to reconstruct/decode the head."""
        return {
            "multilabel": self.multilabel,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "activation_mode": self.activation_mode,
            "dropout_rate": self.dropout_rate,
            "layer_norm": self.layer_norm,
            "loss": self.loss,
            "loss_weight": self.loss_weight,
        }

    def load_meta(self, meta: Dict) -> None:
        """Restore the head's label mappings and configuration from `state_meta`."""
        self.label2id = meta.get("label2id", {})
        self.id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
        self.num_classes = meta.get("num_classes")
        self.hidden_size = meta.get("hidden_size", self.hidden_size)
        self.activation_mode = meta.get("activation_mode", self.activation_mode)
        self.dropout_rate = meta.get("dropout_rate", self.dropout_rate)
        self.layer_norm = meta.get("layer_norm", self.layer_norm)
        self.loss = meta.get("loss", self.loss)
        self.loss_weight = meta.get("loss_weight", self.loss_weight)


@edsnlp.registry.misc.register("eds.single_label_head")
class SingleLabelHead(ClassificationHead):
    """
    Single-label (multi-class) head. Each document has exactly one label for
    this head. Uses cross-entropy or focal loss and decodes via ``argmax``.

    Suitable for the principal diagnosis (`dp`), the mode of care (`mdp`) and
    the DAS count head (whose labels are the integers ``0..K``).
    """

    multilabel = False
    losses = ("ce", "focal")

    def init_loss(self) -> None:
        """Build a cross-entropy (``ce``) or focal (``focal``) loss."""
        weight = self._weight_tensor()
        if self.loss == "focal":
            self.loss_fn = FocalLoss(alpha=weight, gamma=2.0, reduction="mean")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def build_target(self, doc: Doc, attr: str) -> Optional[torch.Tensor]:
        """Map the gold label at ``doc._.<attr>`` to a scalar class-index tensor."""
        label = getattr(doc._, attr, None)
        if label is None:
            return None
        if label in self.label2id:
            label = self.label2id[label]
        elif isinstance(label, str):
            raise ValueError(f"Label '{label}' not in label2id for head '{attr}'.")
        return torch.tensor(int(label), dtype=torch.long)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy / focal loss, moving class weights to the logits device."""
        weight = getattr(self.loss_fn, "weight", None)
        if weight is not None:
            self.loss_fn.weight = weight.to(logits.device)
        alpha = getattr(self.loss_fn, "alpha", None)
        if isinstance(alpha, torch.Tensor):
            self.loss_fn.alpha = alpha.to(logits.device)
        return self.loss_fn(logits, target.to(logits.device))

    def decode(self, logits: torch.Tensor, **ctx) -> List:
        """Return the ``argmax`` label string for each document in the batch."""
        ids = torch.argmax(logits, dim=-1).tolist()
        return [self.id2label.get(i, i) for i in ids]


@edsnlp.registry.misc.register("eds.multi_label_head")
class MultiLabelHead(ClassificationHead):
    """
    Multi-label head. Each document has a (variable-length) set of labels for
    this head. Uses ``BCEWithLogitsLoss`` and decodes either by thresholding
    the per-class probabilities or by selecting the top-``k`` logits, where
    ``k`` is supplied by a companion count head.

    Suitable for associated diagnoses (`das`).

    Parameters
    ----------
    selection : {"threshold", "topk"}, default="threshold"
        How to turn logits into a label set at inference.
    threshold : float, default=0.5
        Probability threshold used when ``selection == "threshold"``.
    count_head : str, optional
        Name of the head whose prediction gives the number of labels ``k`` when
        ``selection == "topk"``.
    """

    multilabel = True
    losses = ("bce",)

    def __init__(
        self,
        *,
        selection: Literal["threshold", "topk"] = "threshold",
        threshold: float = 0.5,
        count_head: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if selection == "topk" and count_head is None:
            raise ValueError(
                "A multi-label head with selection='topk' needs a `count_head` to "
                "tell it how many labels to keep. Either name the head that "
                "predicts that count, or use selection='threshold'."
            )
        self.selection = selection
        self.threshold = threshold
        self.count_head = count_head

    def init_loss(self) -> None:
        """Build a ``BCEWithLogitsLoss``; class weights act as ``pos_weight``."""
        pos_weight = self._weight_tensor()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def build_target(self, doc: Doc, attr: str) -> Optional[torch.Tensor]:
        """Build a multi-hot target vector from the label list at ``doc._.<attr>``."""
        labels = getattr(doc._, attr, None)
        if labels is None:
            return None
        if isinstance(labels, str):
            labels = [labels]
        target = torch.zeros(self.num_classes, dtype=torch.float)
        for label in labels:
            if label in self.label2id:
                target[self.label2id[label]] = 1.0
        return target

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss, moving ``pos_weight`` to the logits device."""
        pos_weight = getattr(self.loss_fn, "pos_weight", None)
        if pos_weight is not None:
            self.loss_fn.pos_weight = pos_weight.to(logits.device)
        return self.loss_fn(logits, target.to(logits.device))

    def _select(
        self, logits: torch.Tensor, selection: str, counts: Optional[List[int]]
    ) -> List[List]:
        """
        Turn logits into a label set per document with the given strategy.

        ``topk`` keeps the ``counts[i]`` highest logits; ``threshold`` keeps all
        labels whose sigmoid probability is ``>= self.threshold``.
        """
        probs = torch.sigmoid(logits)
        n_classes = logits.size(-1)
        results = []
        for i in range(logits.size(0)):
            if selection == "topk":
                k = int(counts[i]) if counts is not None else 0
                k = max(0, min(k, n_classes))
                idx = torch.topk(logits[i], k).indices.tolist() if k > 0 else []
            else:
                idx = (probs[i] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
            results.append([self.id2label.get(j, j) for j in idx])
        return results

    def decode(
        self, logits: torch.Tensor, counts: Optional[List[int]] = None, **ctx
    ) -> List[List]:
        """
        Decode logits into a label set per document using ``self.selection``.

        Parameters
        ----------
        logits : torch.Tensor
            Per-class logits, shape ``(batch_size, num_classes)``.
        counts : List[int], optional
            Number of labels to keep per document, supplied by the companion
            count head when ``selection == "topk"``.
        """
        return self._select(logits, self.selection, counts)

    def decode_alt(
        self, logits: torch.Tensor, counts: Optional[List[int]] = None
    ) -> List[List]:
        """Decode with the *other* selection strategy (for comparison)."""
        other = "threshold" if self.selection == "topk" else "topk"
        return self._select(logits, other, counts)

    def state_meta(self) -> Dict:
        """Extend the base metadata with the selection configuration."""
        meta = super().state_meta()
        meta.update(
            selection=self.selection,
            threshold=self.threshold,
            count_head=self.count_head,
        )
        return meta

    def load_meta(self, meta: Dict) -> None:
        """Restore the base metadata and the selection configuration."""
        super().load_meta(meta)
        self.selection = meta.get("selection", self.selection)
        self.threshold = meta.get("threshold", self.threshold)
        self.count_head = meta.get("count_head", self.count_head)


__all__ = [
    "FocalLoss",
    "ClassificationHead",
    "SingleLabelHead",
    "MultiLabelHead",
]
