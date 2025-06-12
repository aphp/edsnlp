from typing import Any, Dict, Optional, Sequence, Union

# from edsnlp.utils.bindings import Attributes, AttributesArg
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
        label_attr: str = "predicted_class",
        loss_fn=None,
    ):
        super().__init__(nlp, name)
        self.embedding = embedding
        self.label_attr = label_attr
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

        if not hasattr(self.embedding, "output_size"):
            raise ValueError(
                "The embedding component must have an 'output_size' attribute."
            )
        embedding_size = self.embedding.output_size
        self.classifier = torch.nn.Linear(embedding_size, num_classes)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        # Extract embedding for the document
        return {"embedding": self.embedding.preprocess(doc)}

    def preprocess_supervised(self, doc: Doc, label: int) -> Dict[str, Any]:
        # Add label to the preprocessed dict
        d = self.preprocess(doc)
        d["targets"] = torch.tensor(label, dtype=torch.long)
        return d

    def collate(self, batch: Dict[str, Sequence[Any]]) -> DocClassifierBatchInput:
        # Collate embeddings and targets
        embeddings = self.embedding.collate(batch["embedding"])
        batch_input: DocClassifierBatchInput = {"embedding": embeddings}
        if "targets" in batch:
            batch_input["targets"] = torch.stack(batch["targets"])
        return batch_input

    def forward(self, batch: DocClassifierBatchInput) -> DocClassifierBatchOutput:
        # Forward pass: compute logits, loss, and predictions
        embeddings = batch["embedding"]
        logits = self.classifier(embeddings)
        output: DocClassifierBatchOutput = {}
        if "targets" in batch:
            loss = self.loss_fn(logits, batch["targets"])
            output["loss"] = loss
        output["labels"] = torch.argmax(logits, dim=-1)
        return output

    def postprocess(self, docs, results, inputs):
        # Assign predicted label to doc._.<label_attr>
        for doc, result in zip(docs, results):
            setattr(doc._, self.label_attr, int(result["labels"]))
