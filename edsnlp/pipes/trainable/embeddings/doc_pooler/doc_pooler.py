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
    component such as `eds.doc_classifier`.

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
          the document.
        - `"attention"`: weighted sum of the word embeddings, the weights being
          produced by a learned linear attention layer.
        - `"cls"`: use the embedding of the first wordpiece of the underlying
          transformer (the `[CLS]` token) instead of pooling word embeddings. A
          document longer than the transformer `window` is split into several
          windows, whose `[CLS]` embeddings are averaged. This mode is mostly
          meant for short texts, and requires an embedding that exposes a `cls`
          output, such as `eds.transformer`.
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

        # Pool from the flat word embeddings, so that the cost of the pooling
        # does not depend on the length of the longest document of the batch
        embeds = embedding_out["embeddings"]
        flat_embeds = embeds.refold("word").as_tensor()
        device = flat_embeds.device
        words_per_context = torch.as_tensor(embeds.lengths[-1], device=device)
        contexts_per_doc = torch.as_tensor(embeds.lengths[-2], device=device)
        n_docs = len(contexts_per_doc)
        doc_of_context = torch.repeat_interleave(
            torch.arange(n_docs, device=device), contexts_per_doc
        )
        doc_of_word = torch.repeat_interleave(doc_of_context, words_per_context)

        if self.pooling_mode == "attention":
            # Softmax over the words of each document
            scores = self.attention(flat_embeds)
            max_scores = scores.new_full((n_docs, 1), float("-inf"))
            max_scores.index_reduce_(0, doc_of_word, scores.detach(), reduce="amax")
            exp_scores = torch.exp(scores - max_scores[doc_of_word])
            denom = exp_scores.new_zeros(n_docs, 1)
            denom.index_add_(0, doc_of_word, exp_scores)
            pooled = flat_embeds.new_zeros(n_docs, flat_embeds.size(-1))
            pooled.index_add_(0, doc_of_word, flat_embeds * exp_scores)
            pooled = pooled / denom.clamp(min=torch.finfo(denom.dtype).tiny)
        else:
            words_per_doc = torch.zeros(n_docs, dtype=torch.long, device=device)
            words_per_doc.index_add_(0, doc_of_context, words_per_context)
            offsets = torch.cumsum(words_per_doc, 0) - words_per_doc
            pooled = torch.nn.functional.embedding_bag(
                input=torch.arange(len(flat_embeds), device=device),
                weight=flat_embeds,
                offsets=offsets,
                mode=self.pooling_mode,
            )

        return {"embeddings": pooled}
