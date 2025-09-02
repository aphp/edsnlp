import os
import pickle
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Union

import torch
from spacy.tokens import Doc
from typing_extensions import NotRequired, TypedDict

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


class TrainableDocClassifier(
    TorchComponent[DocClassifierBatchOutput, DocClassifierBatchInput],
    BaseComponent,
):
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
        loss_fn=None,
        labels: Optional[Sequence[str]] = None,
        class_weights: Optional[Union[Dict[str, float], str]] = None, 
    ):
        self.label_attr: Attributes = label_attr
        self.label2id = label2id or {}
        self.id2label = id2label or {}
        self.labels = labels
        self.class_weights = class_weights  
        
        super().__init__(nlp, name)
        self.embedding = embedding
        
        self._loss_fn = loss_fn
        self.loss_fn = None

        if not hasattr(self.embedding, "output_size"):
            raise ValueError(
                "The embedding component must have an 'output_size' attribute."
            )
        embedding_size = self.embedding.output_size
        if num_classes:
            self.classifier = torch.nn.Linear(embedding_size, num_classes)

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

    def _load_class_weights_from_file(self, filepath: str) -> Dict[str, int]:
        """Load class weights from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Doc.has_extension(self.label_attr):
            Doc.set_extension(self.label_attr, default={})

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
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
                print("num classes:", len(self.label2id))
                self.classifier = torch.nn.Linear(
                    self.embedding.output_size, len(self.label2id)
                )
        
        weight_tensor = None
        if self.class_weights is not None:
            if isinstance(self.class_weights, str):
                freq_dict = self._load_class_weights_from_file(self.class_weights)
                weight_tensor = self._compute_class_weights(freq_dict)
            elif isinstance(self.class_weights, dict):
                weight_tensor = self._compute_class_weights(self.class_weights)
            
            print(f"Using class weights: {weight_tensor}")
        
        if self._loss_fn is not None:
            self.loss_fn = self._loss_fn
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        
        super().post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        return {"embedding": self.embedding.preprocess(doc)}

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
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
        pooled = self.embedding(batch["embedding"])
        embeddings = pooled["embeddings"]

        logits = self.classifier(embeddings)

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
        labels = results["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        for doc, label in zip(docs, labels):
            if self.id2label and isinstance(label, int):
                label = self.id2label.get(label, label)
            setattr(doc._, self.label_attr, label)
        return docs

    def to_disk(self, path, *, exclude=set()):
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
        data_path = path / "label_attr.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        obj = super().from_disk(path, **kwargs)
        obj.label_attr = data.get("label_attr", "label")
        obj.label2id = data.get("label2id", {})
        obj.id2label = data.get("id2label", {})
        return obj