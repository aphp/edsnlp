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
        "word_to_span_idx": torch.Tensor,
        "span_to_ctx_idx": torch.Tensor,
        "flat_indices": torch.Tensor,
        "offsets": torch.Tensor,
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
word_to_span_idx: torch.LongTensor
    Span index of each token in the flattened span tokens
span_to_ctx_idx: torch.LongTensor
    Sequence/context (cf Embedding spans) index of the spans
flat_indices: torch.LongTensor
    Indices of the tokens in the flattened span tokens
offsets: torch.LongTensor
    Offsets of the spans in the flattened span tokens
stats: Dict[str, int]
    Statistics about the batch, e.g. number of spans
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
    pooling_mode: Literal["max", "sum", "mean", "attention"]
        How word embeddings are aggregated into a single embedding per span:

        - "max": max pooling
        - "sum": sum pooling
        - "mean": mean pooling
        - "attention": attention pooling, where attention scores are computed using a
            linear layer followed by a softmax over the tokens in the span.
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
        pooling_mode: Literal["max", "sum", "mean", "attention"] = "mean",
        activation: Optional[str] = None,
        norm: Optional[str] = None,
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
        self.activation = activation
        self.projector = torch.nn.Sequential()
        if hidden_size is not None:
            self.projector.append(
                torch.nn.Linear(self.embedding.output_size, hidden_size)
            )
        if activation is not None:
            self.projector.append(
                {
                    "relu": torch.nn.ReLU,
                    "gelu": torch.nn.GELU,
                    "silu": torch.nn.SiLU,
                }[activation]()
            )
        if norm is not None:
            self.projector.append(
                {
                    "layernorm": torch.nn.LayerNorm,
                    "batchnorm": torch.nn.BatchNorm1d,
                }[norm](
                    hidden_size
                    if hidden_size is not None
                    else self.embedding.output_size
                )
            )
        if self.pooling_mode in {"attention"}:
            self.attention_scorer = torch.nn.Linear(
                self.embedding.output_size, 1, bias=False
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

        context_indices = []
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
            context_indices.append(contexts_to_idx[ctx[0]])
            begins.append(span.start - start)
            ends.append(span.end - start)
        return {
            "begins": begins,
            "ends": ends,
            "span_to_ctx_idx": context_indices,
            "num_sequences": len(contexts),
            "embedding": self.embedding.preprocess(doc, contexts=contexts, **kwargs),
            "stats": {"spans": len(begins)},
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanPoolerBatchInput:
        embedding_batch = self.embedding.collate(batch["embedding"])
        n_words = embedding_batch["stats"]["words"]
        span_to_ctx_idx = []
        word_to_span_idx = []
        offset_ctx = 0
        offset_span = 0
        flat_indices = []
        offsets = [0]
        for indices, num_sample_contexts, begins, ends in zip(
            batch["span_to_ctx_idx"],
            batch["num_sequences"],
            batch["begins"],
            batch["ends"],
        ):
            span_to_ctx_idx.extend([offset_ctx + idx for idx in indices])
            offset_ctx += num_sample_contexts
            for b, e, ctx_idx in zip(begins, ends, indices):
                offset_word = n_words * ctx_idx
                word_to_span_idx.extend([offset_span] * (e - b))
                flat_indices.extend(range(offset_word + b, offset_word + e))
                offsets.append(len(flat_indices))
                offset_span += 1
        offsets = offsets[:-1]

        begins = ft.as_folded_tensor(
            batch["begins"],
            data_dims=("span",),
            full_names=("sample", "span"),
            dtype=torch.long,
        )
        ends = ft.as_folded_tensor(
            batch["ends"],
            data_dims=("span",),
            full_names=("sample", "span"),
            dtype=torch.long,
        )
        collated: SpanPoolerBatchInput = {
            "embedding": embedding_batch,
            "begins": begins,
            "ends": ends,
            "flat_indices": torch.as_tensor(flat_indices),  # (num_span_tokens,)
            "offsets": torch.as_tensor(offsets),  # (num_spans,)
            "word_to_span_idx": torch.as_tensor(word_to_span_idx),  # (num_span_tokens,)
            "span_to_ctx_idx": torch.as_tensor(span_to_ctx_idx),  # (num_spans,)
            "stats": {"spans": sum(batch["stats"]["spans"])},
        }
        return collated

    def _pool_spans(self, flat_embeds, word_to_span_idx, offsets):
        dev = offsets.device
        dim = flat_embeds.size(-1)
        n_spans = len(offsets)

        if self.pooling_mode == "attention":
            weights = self.attention_scorer(flat_embeds)
            # compute max for softmax stability
            max_weights = torch.full((n_spans, 1), float("-inf"), device=dev)
            max_weights.index_reduce_(0, word_to_span_idx, weights, reduce="amax")
            # softmax numerator
            exp_scores = torch.exp(weights - max_weights[word_to_span_idx])
            # softmax denominator
            denom = torch.zeros((n_spans, 1), device=dev)
            denom.index_add_(0, word_to_span_idx, exp_scores)
            # softmax output = embeds * weight num / weight denom
            weighted_embeds = flat_embeds * exp_scores / denom[word_to_span_idx]
            span_embeds = torch.zeros((n_spans, dim), device=dev)
            span_embeds.index_add_(0, word_to_span_idx, weighted_embeds)
        else:
            span_embeds = torch.nn.functional.embedding_bag(  # type: ignore
                input=torch.arange(len(flat_embeds), device=dev),
                weight=flat_embeds,
                offsets=offsets,
                mode=self.pooling_mode,
            )
        span_embeds = self.feed_forward(span_embeds)
        return span_embeds

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
        n_spans = len(batch["begins"])
        word_to_span_idx = batch["word_to_span_idx"]
        offsets = batch["offsets"]
        flat_indices = batch["flat_indices"]

        if n_spans == 0:
            span_embeds = torch.empty(0, self.output_size, device=offsets.dev)
            return {
                "embeddings": batch["begins"].with_data(span_embeds),
            }

        embeds = self.embedding(batch["embedding"])["embeddings"]
        embeds = embeds.refold(["context", "word"])
        flat_embeds = embeds.view(-1, embeds.size(-1))[flat_indices]
        span_embeds = self._pool_spans(
            flat_embeds,
            word_to_span_idx,
            offsets,
        )
        return {
            "embeddings": batch["begins"].with_data(span_embeds),
        }
