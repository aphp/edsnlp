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
        "mask": torch.Tensor,
        "stats": Dict[str, Any],
    },
)

DocPoolerBatchOutput = TypedDict(
    "DocPoolerBatchOutput",
    {
        "embeddings": torch.Tensor,
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
        pooling_mode: Literal["max", "sum", "mean", "cls", "attention"] = "mean",
    ):
        super().__init__(nlp, name)
        self.embedding = embedding
        self.pooling_mode = pooling_mode
        self.output_size = embedding.output_size

        # Add attention layer if needed
        if pooling_mode == "attention":
            self.attention = torch.nn.Linear(self.output_size, 1)

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
        """
        Forward pass: compute document embeddings using the selected pooling strategy
        """
        embeds = self.embedding(batch["embedding"])["embeddings"].refold(
            "context", "word"
        )
        device = embeds.device

        if self.pooling_mode == "cls":
            pooled = self.embedding(batch["embedding"])["cls"].to(device)
            return {"embeddings": pooled}

        mask = embeds.mask

        if self.pooling_mode == "attention":
            attention_weights = self.attention(embeds)  # (batch_size, seq_len, 1)
            attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)

            attention_weights = attention_weights.masked_fill(~mask, float("-inf"))

            attention_weights = torch.softmax(attention_weights, dim=1)

            attention_weights = attention_weights.unsqueeze(
                -1
            )  # (batch_size, seq_len, 1)
            pooled = (embeds * attention_weights).sum(dim=1)  # (batch_size, embed_dim)

        else:
            mask_expanded = mask.unsqueeze(-1)
            masked_embeds = embeds * mask_expanded
            sum_embeds = masked_embeds.sum(dim=1)

            if self.pooling_mode == "mean":
                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = sum_embeds / valid_counts
            elif self.pooling_mode == "max":
                masked_embeds = embeds.masked_fill(~mask_expanded, float("-inf"))
                pooled, _ = masked_embeds.max(dim=1)
            elif self.pooling_mode == "sum":
                pooled = sum_embeds
            else:
                raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return {"embeddings": pooled}
