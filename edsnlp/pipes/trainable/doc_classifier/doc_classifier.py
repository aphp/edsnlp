import os
import pickle
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Union

import pandas as pd
import torch
import torch.nn as nn
from spacy.tokens import Doc
from typing_extensions import Literal, NotRequired, TypedDict

import edsnlp
from edsnlp.core.pipeline import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, TorchComponent
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    WordContextualizerComponent,
    WordEmbeddingComponent,
)
from edsnlp.utils.bindings import Attributes

DocClassifierBatchInput = TypedDict(
    "DocClassifierBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
    },
)

DocClassifierBatchOutput = TypedDict(
    "DocClassifierBatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "labels": Optional[torch.Tensor],
    },
)


@edsnlp.registry.misc.register("focal_loss")
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.

    Parameters
    ----------
    alpha : torch.Tensor or float, optional
        Class weights. If None, no weighting is applied
    gamma : float, default=2.0
        Focusing parameter. Higher values give more weight to hard examples
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TrainableDocClassifier(
    TorchComponent[DocClassifierBatchOutput, DocClassifierBatchInput],
    BaseComponent,
):
    """A trainable document classifier that uses embeddings to classify documents."""

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "doc_classifier",
        *,
        embedding: Union[WordEmbeddingComponent, WordContextualizerComponent],
        num_classes: Optional[int] = None,
        label_attr: str = "label",
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        loss: Literal["ce", "focal"] = "ce",
        labels: Optional[str] = None,
        class_weights: Optional[str] = None,
        hidden_size: Optional[int] = None,
        activation_mode: Literal["relu", "gelu", "silu"] = "relu",
        dropout_rate: Optional[float] = 0.0,
        layer_norm: Optional[bool] = False,
    ):
        self.num_classes = num_classes
        self.label_attr: Attributes = label_attr
        self.label2id = label2id or {}
        self.id2label = id2label or {}
        if labels:
            self.labels = pd.read_pickle(labels)
            self.num_classes = len(self.labels)
        if class_weights:
            self.class_weights = pd.read_pickle(class_weights)
        self.hidden_size = hidden_size
        self.activation_mode = activation_mode
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm

        super().__init__(nlp, name)
        self.embedding = embedding

        self.loss = loss

        if not hasattr(self.embedding, "output_size"):
            raise ValueError(
                "The embedding component must have an 'output_size' attribute."
            )
        self.embedding_size = self.embedding.output_size
        if self.num_classes:
            self.build_classifier()

    def build_classifier(self):
        """Build classification head"""
        if self.hidden_size:
            self.hidden_layer = torch.nn.Linear(self.embedding_size, self.hidden_size)
            self.activation = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[
                self.activation_mode
            ]
            if self.layer_norm:
                self.norm = nn.LayerNorm(self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.classifier = torch.nn.Linear(self.hidden_size, self.num_classes)
        else:
            self.classifier = torch.nn.Linear(self.embedding_size, self.num_classes)

    def _compute_class_weights(self, freq_dict: Dict[str, int]) -> torch.Tensor:
        """
        Compute class weights from frequency dictionary.
        Uses inverse frequency weighting: weight = 1 / frequency
        """
        total_samples = sum(freq_dict.values())

        weights = torch.zeros(len(self.label2id))

        for label, freq in freq_dict.items():
            if label in self.label2id:
                weight = total_samples / (len(self.label2id) * freq)
                weights[self.label2id[label]] = weight

        return weights

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Doc.has_extension(self.label_attr):
            Doc.set_extension(self.label_attr, default={})

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        print("post_init")
        if not self.label2id:
            if self.labels is not None:
                labels = set(self.labels)
            else:
                labels = set()
                for doc in gold_data:
                    label = getattr(doc._, self.label_attr, None)
                    if isinstance(label, str):
                        labels.add(label)
            if labels:
                self.label2id = {}
                self.id2label = {}
                for i, label in enumerate(labels):
                    self.label2id[label] = i
                    self.id2label[i] = label
                self.num_classes = len(self.label2id)
                print("num classes:", self.num_classes)
                self.build_classifier()
        print("label2id fini")
        weight_tensor = None
        if self.class_weights is not None:
            weight_tensor = self._compute_class_weights(self.class_weights)
            print(f"Using class weights: {weight_tensor}")
        print("weight tensor fini")
        if self.loss == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        elif self.loss == "focal":
            self.loss_fn = FocalLoss(alpha=weight_tensor, gamma=2.0, reduction="mean")
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        print("loss finie")
        super().post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        return {"embedding": self.embedding.preprocess(doc)}

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        """Preprocess document with target labels for training."""
        preps = self.preprocess(doc)
        label = getattr(doc._, self.label_attr, None)
        if label is None:
            raise ValueError(
                f"Document does not have a gold label in 'doc._.{self.label_attr}'"
            )
        if isinstance(label, str) and self.label2id:
            if label not in self.label2id:
                raise ValueError(f"Label '{label}' not in label2id mapping.")
            label = self.label2id[label]
        return {
            **preps,
            "targets": torch.tensor(label, dtype=torch.long),
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> DocClassifierBatchInput:
        embeddings = self.embedding.collate(batch["embedding"])
        batch_input: DocClassifierBatchInput = {"embedding": embeddings}
        if "targets" in batch:
            batch_input["targets"] = torch.stack(batch["targets"])
        return batch_input

    def forward(self, batch: DocClassifierBatchInput) -> DocClassifierBatchOutput:
        """
        Forward pass: compute embeddings, classify, and calculate loss
        if targets provided.
        """
        pooled = self.embedding(batch["embedding"])
        x = pooled["embeddings"]
        if self.hidden_size:
            x = self.hidden_layer(x)
            x = self.activation(x)
            if self.layer_norm:
                x = self.norm(x)
            x = self.dropout(x)
        logits = self.classifier(x)

        output: DocClassifierBatchOutput = {}
        if "targets" in batch:
            loss = self.loss_fn(logits, batch["targets"])
            output["loss"] = loss
            output["labels"] = None
        else:
            output["loss"] = None
            output["labels"] = torch.argmax(logits, dim=-1)
        return output

    def postprocess(self, docs, results, input):
        """Postprocess predictions by assigning labels to documents."""
        labels = results["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        for doc, label in zip(docs, labels):
            if self.id2label and isinstance(label, int):
                label = self.id2label.get(label, label)
            setattr(doc._, self.label_attr, label)
        return docs

    def to_disk(self, path, *, exclude=set()):
        """Save classifier state to disk."""
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        exclude.add(repr_id)
        os.makedirs(path, exist_ok=True)
        data_path = path / "label_attr.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "label_attr": self.label_attr,
                    "label2id": self.label2id,
                    "id2label": self.id2label,
                },
                f,
            )
        return super().to_disk(path, exclude=exclude)

    @classmethod
    def from_disk(cls, path, **kwargs):
        """Load classifier from disk."""
        data_path = path / "label_attr.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        obj = super().from_disk(path, **kwargs)
        obj.label_attr = data.get("label_attr", "label")
        obj.label2id = data.get("label2id", {})
        obj.id2label = data.get("id2label", {})
        return obj
