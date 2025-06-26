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
        num_classes: int,
        label_attr: str = "label",
        loss_fn=None,
    ):
        self.label_attr: Attributes = label_attr
        super().__init__(nlp, name)
        self.embedding = embedding
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

        if not hasattr(self.embedding, "output_size"):
            raise ValueError(
                "The embedding component must have an 'output_size' attribute."
            )
        embedding_size = self.embedding.output_size
        self.classifier = torch.nn.Linear(embedding_size, num_classes)

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Doc.has_extension(self.label_attr):
            Doc.set_extension(self.label_attr, default={})

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
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
            setattr(doc._, self.label_attr, label)
            # doc._.label = label
        return docs

    def to_disk(self, path, *, exclude=set()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        exclude.add(repr_id)
        os.makedirs(path, exist_ok=True)
        data_path = path / "label_attr.pkl"
        with open(data_path, "wb") as f:
            pickle.dump({"label_attr": self.label_attr}, f)
        return super().to_disk(path, exclude=exclude)
