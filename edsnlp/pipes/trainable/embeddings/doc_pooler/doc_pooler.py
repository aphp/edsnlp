from typing import Any, Dict, Optional

import torch
from spacy.tokens import Doc
from typing_extensions import Literal, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent

DocPoolerBatchInput = TypedDict(
    "DocPoolerBatchInput",
    {
        "embedding": BatchInput,
        "mask": torch.Tensor,  # shape: (batch_size, seq_len)
        "stats": Dict[str, Any],
    },
)

DocPoolerBatchOutput = TypedDict(
    "DocPoolerBatchOutput",
    {
        "embeddings": torch.Tensor,  # shape: (batch_size, embedding_dim)
    },
)


class DocPooler(WordEmbeddingComponent, BaseComponent):
    """
    Pools word embeddings over the entire document to produce
    a single embedding per doc.

    Parameters
    ----------
    nlp: Pipeline
        The pipeline object
    name: str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    pooling_mode: Literal["max", "sum", "mean"]
        How word embeddings are aggregated into a single embedding per document.
    hidden_size : Optional[int]
        The size of the hidden layer. If None, no projection is done.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "document_pooler",
        *,
        embedding: WordEmbeddingComponent,
        pooling_mode: Literal["max", "sum", "mean", "cls"] = "mean",
        hidden_size: Optional[int] = None,
    ):
        super().__init__(nlp, name)
        self.embedding = embedding
        self.pooling_mode = pooling_mode
        self.output_size = embedding.output_size if hidden_size is None else hidden_size
        self.projector = (
            torch.nn.Linear(self.embedding.output_size, hidden_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )

    def feed_forward(self, doc_embed: torch.Tensor) -> torch.Tensor:
        return self.projector(doc_embed)

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        embedding_out = self.embedding.preprocess(doc, **kwargs)
        return {
            "embedding": embedding_out,
            "stats": {"doc_length": len(doc)},
        }

    def collate(self, batch: Dict[str, Any]) -> DocPoolerBatchInput:
        embedding_batch = self.embedding.collate(batch["embedding"])
        stats = batch["stats"]
        return {
            "embedding": embedding_batch,
            "stats": {
                "doc_length": sum(stats["doc_length"])
            },  # <-- sum(...) pour aggréger les comptes par doc en un compte par batch
        }

    def forward(self, batch: DocPoolerBatchInput) -> DocPoolerBatchOutput:
        device = next(self.parameters()).device

        embeds = self.embedding(batch["embedding"])["embeddings"]
        device = embeds.device

        if self.pooling_mode == "mean":
            pooled = embeds.mean(dim=1)
        elif self.pooling_mode == "max":
            pooled = embeds.max(dim=1).values
        elif self.pooling_mode == "sum":
            pooled = embeds.sum(dim=1) / embeds.size(1)
        elif self.pooling_mode == "cls":
            pooled = self.embedding(batch["embedding"])["cls"].to(device)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        pooled = self.feed_forward(pooled)
        return {"embeddings": pooled}
