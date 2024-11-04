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
        "ends": ft.FoldedTensor,
        "sequence_idx": torch.Tensor,
        "stats": TypedDict("SpanPoolerBatchStats", {"spans": int}),
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
begins: torch.LongTensor
    Begin offsets of the spans
ends: torch.LongTensor
    End offsets of the spans
sequence_idx: torch.LongTensor
    Sequence (cf Embedding spans) index of the spans
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
        sequence_idx = []
        offset = 0
        for indices, seq_length in zip(batch["sequence_idx"], batch["num_sequences"]):
            sequence_idx.extend([offset + idx for idx in indices])
            offset += seq_length

        collated: SpanPoolerBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "begins": ft.as_folded_tensor(
                batch["begins"],
                data_dims=("span",),
                full_names=("sample", "span"),
                dtype=torch.long,
            ),
            "ends": ft.as_folded_tensor(
                batch["ends"],
                data_dims=("span",),
                full_names=("sample", "span"),
                dtype=torch.long,
            ),
            "sequence_idx": torch.as_tensor(sequence_idx),
            "stats": {"spans": sum(batch["stats"]["spans"])},
        }
        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanPoolerBatchInput) -> SpanPoolerBatchOutput:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans
        If labels are predicted, they are assigned to the `additional_outputs`
        dictionary.

        Parameters
        ----------
        batch: SpanPoolerBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        device = next(self.parameters()).device
        if len(batch["begins"]) == 0:
            span_embeds = torch.empty(0, self.output_size, device=device)
            return {
                "embeddings": batch["begins"].with_data(span_embeds),
            }

        embeds = self.embedding(batch["embedding"])["embeddings"]
        _, n_words, dim = embeds.shape
        device = embeds.device

        flat_begins = n_words * batch["sequence_idx"] + batch["begins"].as_tensor()
        flat_ends = n_words * batch["sequence_idx"] + batch["ends"].as_tensor()
        flat_embeds = embeds.view(-1, dim)
        flat_indices = torch.cat(
            [
                torch.arange(b, e, device=device)
                for b, e in zip(flat_begins.cpu().tolist(), flat_ends.cpu().tolist())
            ]
        ).to(device)
        offsets = (flat_ends - flat_begins).cumsum(0).roll(1)
        offsets[0] = 0
        span_embeds = torch.nn.functional.embedding_bag(  # type: ignore
            input=flat_indices,
            weight=flat_embeds,
            offsets=offsets,
            mode=self.pooling_mode,
        )
        span_embeds = self.feed_forward(span_embeds)

        return {
            "embeddings": batch["begins"].with_data(span_embeds),
        }
