from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from spacy.tokens import Doc, Span
from typing_extensions import Literal, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    DocEmbeddingBatchOutput,
    DocEmbeddingComponent,
    WordEmbeddingComponent,
)
from edsnlp.pipes.trainable.layers.pooling import Pooling

DocPoolerBatchInput = TypedDict(
    "DocPoolerBatchInput",
    {
        "embedding": BatchInput,
        "stats": Dict[str, Any],
    },
)


class DocPooler(DocEmbeddingComponent, BaseComponent):
    """
    The `eds.doc_pooler` component aggregates the word embeddings of a document
    into a single embedding per document, to be consumed by a document-level
    component such as `eds.doc_classifier`. When the document is split into
    several contexts, they are pooled together into a single embedding.

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
    pooling_mode: Literal["max", "sum", "mean", "attention"]
        How the embeddings are aggregated into a single embedding per document:

        - `"mean"`, `"max"`, `"sum"`: element-wise reduction over the embeddings
          of the document
        - `"attention"`: weighted sum of the embeddings of the document, the weights
          being a softmax of scores given by a learned linear layer
    use_cls: bool
        Pool the embeddings of the first wordpiece of each transformer window (the
        `[CLS]` token) instead of the word embeddings. A document longer than the
        transformer `window` is split into several windows, whose `[CLS]` embeddings
        are aggregated with `pooling_mode`. This is mostly meant for short texts, and
        requires an embedding that returns a `cls` output, such as `eds.transformer`.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "document_pooler",
        *,
        embedding: WordEmbeddingComponent,
        pooling_mode: Literal["max", "sum", "mean", "attention"] = "mean",
        use_cls: bool = False,
    ):
        super().__init__(nlp, name)
        self.embedding = embedding
        self.pooling_mode = pooling_mode
        self.use_cls = use_cls
        self.output_size = embedding.output_size
        self.pooling = Pooling(self.output_size, pooling_mode)

    def preprocess(
        self,
        doc: Doc,
        *,
        contexts: Optional[Sequence[Span]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        embedding_out = self.embedding.preprocess(doc, contexts=contexts, **kwargs)
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

    def forward(self, batch: DocPoolerBatchInput) -> DocEmbeddingBatchOutput:
        """
        Forward pass: compute document embeddings using the selected pooling strategy
        """
        embedding_out = self.embedding(batch["embedding"])
        embeds = embedding_out["embeddings"]
        device = embeds.device
        contexts_per_doc = torch.as_tensor(embeds.lengths[-2], device=device)
        n_docs = len(contexts_per_doc)
        doc_of_context = torch.repeat_interleave(
            torch.arange(n_docs, device=device), contexts_per_doc
        )

        if self.use_cls:
            if "cls" not in embedding_out:
                raise ValueError(
                    f"`use_cls=True` requires an embedding that returns a 'cls' "
                    f"output, which {type(self.embedding).__name__} does not. "
                    f"Use `eds.transformer` as the embedding, or pool the word "
                    f"embeddings instead."
                )
            flat_embeds = embedding_out["cls"]
            doc_of_item = doc_of_context[embedding_out["cls_contexts"]]
        else:
            flat_embeds = embeds.refold("word").as_tensor()
            words_per_context = torch.as_tensor(embeds.lengths[-1], device=device)
            doc_of_item = torch.repeat_interleave(doc_of_context, words_per_context)

        # Pool the flat embeddings of each document, so that the cost of the pooling
        # does not depend on the length of the longest document of the batch
        indices = torch.argsort(doc_of_item, stable=True)
        items_per_doc = torch.bincount(doc_of_item, minlength=n_docs)
        offsets = torch.cumsum(items_per_doc, 0) - items_per_doc
        return {"embeddings": self.pooling(flat_embeds, indices, offsets)}
