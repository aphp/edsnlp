from __future__ import annotations

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
    The `eds.doc_pooler` component aggregates the word embeddings of a document
    into a single embedding per document, to be consumed by a document-level
    component such as [`eds.doc_classifier`](/pipes/trainable/doc-classifier).

    Examples
    --------
    ```{ .python }
    import edsnlp, edsnlp.pipes as eds

    embedding = eds.doc_pooler(
        pooling_mode="attention",
        embedding=eds.transformer(
            model="hf-internal-testing/tiny-random-bert",
            window=128,
            stride=96,
        ),
    )
    ```

    Parameters
    ----------
    nlp: Pipeline
        The pipeline object
    name: str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    pooling_mode: Literal["max", "sum", "mean", "cls", "attention"]
        How word embeddings are aggregated into a single embedding per document:

        - `"mean"`, `"max"`, `"sum"`: element-wise reduction over the words of
          the document, ignoring padding.
        - `"attention"`: weighted sum of the word embeddings, the weights being
          produced by a learned linear attention layer.
        - `"cls"`: use the embedding of the first wordpiece of the underlying
          transformer (the `[CLS]` token) instead of pooling word embeddings.

        !!! warning "`cls` and long documents"

            The `"cls"` mode returns one vector per *context*, and a document
            longer than the transformer `window` is split into several
            overlapping contexts. It should therefore only be used when
            documents are known to fit in a single window. It also requires the
            embedding to expose a `cls` output, which only `eds.transformer`
            does.
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
            # sum(...) aggregates the per-doc counts into a single batch count
            "stats": {"doc_length": sum(stats["doc_length"])},
        }

    def forward(self, batch: DocPoolerBatchInput) -> DocPoolerBatchOutput:
        """
        Forward pass: compute document embeddings using the selected pooling strategy
        """
        embedding_out = self.embedding(batch["embedding"])

        if self.pooling_mode == "cls":
            if "cls" not in embedding_out:
                raise ValueError(
                    f"`pooling_mode='cls'` requires an embedding that returns a "
                    f"'cls' output, which {type(self.embedding).__name__} does not. "
                    f"Use `eds.transformer` as the embedding, or pool the word "
                    f"embeddings with 'mean', 'max', 'sum' or 'attention' instead."
                )
            return {"embeddings": embedding_out["cls"]}

        embeds = embedding_out["embeddings"].refold("context", "word")
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
            else:  # pragma: no cover
                raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return {"embeddings": pooled}
