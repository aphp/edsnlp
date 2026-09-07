from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import foldedtensor as ft
import torch
from confit import VisibleDeprecationWarning
from spacy.tokens import Doc, Span
from typing_extensions import Literal, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
    WordEmbeddingComponent,
)
from edsnlp.utils.filter import align_spans

SpanPoolerBatchInput = TypedDict(
    "SpanPoolerBatchInput",
    {
        "embedding": BatchInput,
        "begins": ft.FoldedTensor,
        "word_indices": torch.Tensor,
        "sequence_idx": torch.Tensor,
        "offsets": torch.Tensor,
        "stats": TypedDict("SpanPoolerBatchStats", {"spans": int}),
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
begins: torch.LongTensor
    Begin offsets of the spans
word_indices: torch.LongTensor
    Word positions within each context for all pooled spans
sequence_idx: torch.LongTensor
    Context index for each pooled word
offsets: torch.LongTensor
    Start of each span in the pooled word indices
"""

SpanPoolerBatchOutput = TypedDict(
    "SpanPoolerBatchOutput",
    {
        "embeddings": ft.FoldedTensor,
    },
)


class SpanPooler(SpanEmbeddingComponent, BaseComponent):
    """
    The `eds.span_pooler` component is a trainable span embedding component. It
    generates span embeddings from a word embedding component and a span getter. It can
    be used to train a span classifier, as in `eds.span_classifier`.

    Parameters
    ----------
    nlp: PipelineProtocol
        The pipeline object
    name: str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    pooling_mode: Literal["max", "sum", "mean"]
        How word embeddings are aggregated into a single embedding per span.
    hidden_size : Optional[int]
        The size of the hidden layer. If None, no projection is done and the output
        of the span pooler is used directly.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "span_pooler",
        *,
        embedding: WordEmbeddingComponent,
        pooling_mode: Literal["max", "sum", "mean"] = "mean",
        hidden_size: Optional[int] = None,
        span_getter: Any = None,
    ):
        if span_getter is not None:
            warnings.warn(
                "The `span_getter` parameter of the `eds.span_pooler` component is "
                "deprecated. Please use the `span_getter` parameter of the "
                "`eds.span_classifier` or `eds.span_linker` components instead.",
                VisibleDeprecationWarning,
            )
        sub_span_getter = getattr(embedding, "span_getter", None)
        if sub_span_getter is not None and span_getter is None:  # pragma: no cover
            self.span_getter = sub_span_getter
        sub_context_getter = getattr(embedding, "context_getter", None)
        if sub_context_getter is not None:  # pragma: no cover
            self.context_getter = sub_context_getter

        self.output_size = embedding.output_size if hidden_size is None else hidden_size

        super().__init__(nlp, name)

        self.pooling_mode = pooling_mode
        self.span_getter = span_getter
        self.embedding = embedding
        self.projector = (
            torch.nn.Linear(self.embedding.output_size, hidden_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )

    def feed_forward(self, span_embeds: torch.Tensor) -> torch.Tensor:
        return self.projector(span_embeds)

    def preprocess(
        self,
        doc: Doc,
        *,
        spans: Optional[Sequence[Span]] = None,
        contexts: Optional[Sequence[Span]] = None,
        pre_aligned: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        contexts = contexts if contexts is not None else [doc[:]]

        sequence_idx = []
        begins = []
        ends = []

        contexts_to_idx = {span: i for i, span in enumerate(contexts)}
        assert not pre_aligned or len(spans) == len(contexts), (
            "When `pre_aligned` is True, the number of spans and contexts must be the "
            "same."
        )
        aligned_contexts = (
            [[c] for c in contexts]
            if pre_aligned
            else align_spans(contexts, spans, sort_by_overlap=True)
        )
        for i, (span, ctx) in enumerate(zip(spans, aligned_contexts)):
            if len(ctx) == 0 or ctx[0].start > span.start or ctx[0].end < span.end:
                raise Exception(
                    f"Span {span.text!r} is not included in at least one embedding "
                    f"span: {[s.text for s in ctx]}"
                )
            start = ctx[0].start
            sequence_idx.append(contexts_to_idx[ctx[0]])
            begins.append(span.start - start)
            ends.append(span.end - start)
        return {
            "begins": begins,
            "ends": ends,
            "sequence_idx": sequence_idx,
            "num_sequences": len(contexts),
            "embedding": self.embedding.preprocess(doc, contexts=contexts, **kwargs),
            "stats": {"spans": len(begins)},
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanPoolerBatchInput:
        """
        Prepare ragged word indices on CPU for the pooled embeddings in forward
        """
        word_indices = []
        sequence_idx = []
        offsets = []
        offset = 0
        for begins, ends, indices, seq_length in zip(
            batch["begins"],
            batch["ends"],
            batch["sequence_idx"],
            batch["num_sequences"],
        ):
            for begin, end, idx in zip(begins, ends, indices):
                offsets.append(len(word_indices))
                word_indices.extend(range(begin, end))
                sequence_idx.extend([offset + idx] * (end - begin))
            offset += seq_length

        collated: SpanPoolerBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "begins": ft.as_folded_tensor(
                batch["begins"],
                data_dims=("span",),
                full_names=("sample", "span"),
                dtype=torch.long,
            ),
            "word_indices": torch.as_tensor(word_indices, dtype=torch.long),
            "sequence_idx": torch.as_tensor(sequence_idx, dtype=torch.long),
            "offsets": torch.as_tensor(offsets, dtype=torch.long),
            "stats": {"spans": sum(batch["stats"]["spans"])},
        }
        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanPoolerBatchInput) -> SpanPoolerBatchOutput:
        """
        Pool context word embeddings into the span embeddings used by classifiers

        Parameters
        ----------
        batch: SpanPoolerBatchInput
            The input batch

        Returns
        -------
        SpanPoolerBatchOutput
            One embedding per span with sample and span dimensions preserved
        """
        device = next(self.parameters()).device
        if len(batch["begins"]) == 0:
            span_embeds = torch.empty(0, self.output_size, device=device)
            return {
                "embeddings": batch["begins"].with_data(span_embeds),
            }

        embeds = self.embedding(batch["embedding"])["embeddings"]
        _, n_words, dim = embeds.shape
        flat_embeds = embeds.view(-1, dim)
        # The embedding shape supplies the padded context width for each word index
        flat_indices = n_words * batch["sequence_idx"] + batch["word_indices"]
        span_embeds = torch.nn.functional.embedding_bag(  # type: ignore
            input=flat_indices,
            weight=flat_embeds,
            offsets=batch["offsets"],
            mode=self.pooling_mode,
        )
        span_embeds = self.feed_forward(span_embeds)

        return {
            "embeddings": batch["begins"].with_data(span_embeds),
        }
