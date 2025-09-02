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
        "span_begins": ft.FoldedTensor,
        "span_ends": ft.FoldedTensor,
        "span_contexts": ft.FoldedTensor,
        "item_indices": torch.LongTensor,
        "span_offsets": torch.LongTensor,
        "span_indices": torch.LongTensor,
        "stats": Dict[str, int],
    },
)
"""
Attributes
----------
embedding: BatchInput
    The input batch for the word embedding component
span_begins: ft.FoldedTensor
    Begin offsets of the spans
span_ends: ft.FoldedTensor
    End offsets of the spans
span_contexts: ft.FoldedTensor
    Sequence/context index of the spans
item_indices: torch.LongTensor
    Indices of the span's tokens in the tokens embedding output
span_offsets: torch.LongTensor
    Offsets of the spans in the flattened span tokens
span_indices: torch.LongTensor
    Span index of each token in the flattened span tokens
stats: Dict[str, int]
    Statistics about the batch, e.g. number of spans
"""

SpanPoolerBatchOutput = TypedDict(
    "SpanPoolerBatchOutput",
    {
        "embeddings": ft.FoldedTensor,
    },
)
"""
Attributes
----------
embeddings: ft.FoldedTensor
    The output span embeddings, with foldable dimensions ("sample", "span")
"""


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
        activation: Optional[Literal["relu", "gelu", "silu"]] = None,
        norm: Optional[Literal["layernorm", "batchnorm"]] = None,
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
        spans: Optional[Sequence[Span]],
        contexts: Optional[Sequence[Span]] = None,
        pre_aligned: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if contexts is None:
            contexts = [doc[:]] * len(spans)
            pre_aligned = True

        context_indices = []
        begins = []
        ends = []

        contexts_to_idx = {}
        for ctx in contexts:
            if ctx not in contexts_to_idx:
                contexts_to_idx[ctx] = len(contexts_to_idx)
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
            "embedding": self.embedding.preprocess(
                doc, contexts=list(contexts_to_idx), **kwargs
            ),
            "stats": {"spans": len(begins)},
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanPoolerBatchInput:
        embedding_batch = self.embedding.collate(batch["embedding"])
        embed_structure = embedding_batch["out_structure"]
        ft_kw = dict(
            data_dims=("span",),
            full_names=("sample", "span"),
            dtype=torch.long,
        )
        begins = ft.as_folded_tensor(batch["begins"], **ft_kw)
        ends = ft.as_folded_tensor(batch["ends"], **ft_kw)
        span_to_ctx_idx = []
        total_num_ctx = 0
        for i, (ctx_indices, num_ctx) in enumerate(
            zip(batch["span_to_ctx_idx"], embed_structure["context"])
        ):
            span_to_ctx_idx.append([idx + total_num_ctx for idx in ctx_indices])
            total_num_ctx += num_ctx
        flat_span_to_ctx_idx = ft.as_folded_tensor(span_to_ctx_idx, **ft_kw)
        item_indices, span_offsets, span_indices = embed_structure.make_indices_ranges(
            begins=(flat_span_to_ctx_idx, begins),
            ends=(flat_span_to_ctx_idx, ends),
            indice_dims=("context", "word"),
        )

        collated: SpanPoolerBatchInput = {
            "embedding": embedding_batch,
            "span_begins": begins,
            "span_ends": ends,
            "span_contexts": flat_span_to_ctx_idx,
            "item_indices": item_indices,
            "span_offsets": begins.with_data(span_offsets),
            "span_indices": span_indices,
            "stats": {"spans": sum(batch["stats"]["spans"])},
        }
        return collated

    def _pool_spans(self, embeds, span_indices, span_offsets, item_indices=None):
        dev = span_offsets.device
        dim = embeds.size(-1)
        embeds = embeds.as_tensor().view(-1, dim)
        N = span_offsets.numel()  # number of spans

        if self.pooling_mode == "attention":
            if item_indices is not None:
                embeds = embeds[item_indices]
            weights = self.attention_scorer(embeds)
            # compute max for softmax stability
            dtype = weights.dtype
            max_weights = torch.full((N, 1), float("-inf"), device=dev, dtype=dtype)
            max_weights.index_reduce_(0, span_indices, weights, reduce="amax")
            # softmax numerator
            exp_scores = torch.exp(weights - max_weights[span_indices])
            # softmax denominator
            denom = torch.zeros((N, 1), device=dev, dtype=exp_scores.dtype)
            denom.index_add_(0, span_indices, exp_scores)
            # softmax output = embeds * weight num / weight denom
            weighted_embeds = embeds * exp_scores / denom[span_indices]
            span_embeds = torch.zeros((N, dim), device=dev, dtype=embeds.dtype)
            span_embeds.index_add_(0, span_indices, weighted_embeds)
            span_embeds = span_offsets.with_data(span_embeds)
        else:
            span_embeds = torch.nn.functional.embedding_bag(  # type: ignore
                input=torch.arange(len(embeds), device=dev)
                if item_indices is None
                else item_indices,
                weight=embeds,
                offsets=span_offsets,
                mode=self.pooling_mode,
            )
        span_embeds = self.feed_forward(span_embeds)
        return span_embeds

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanPoolerBatchInput) -> SpanPoolerBatchOutput:
        """
        Forward pass of the component, returns span embeddings.

        Parameters
        ----------
        batch: SpanPoolerBatchInput
            The input batch

        Returns
        -------
        SpanPoolerBatchOutput
        """
        if len(batch["span_begins"]) == 0:
            return {
                "embeddings": batch["span_begins"].with_data(
                    torch.empty(
                        0,
                        self.output_size,
                        device=batch["item_indices"].device,
                    )
                ),
            }

        embeds = self.embedding(batch["embedding"])["embeddings"]
        span_embeds = self._pool_spans(
            embeds,
            span_indices=batch["span_indices"],
            span_offsets=batch["span_offsets"],
            item_indices=batch["item_indices"],
        )
        return {
            "embeddings": batch["span_begins"].with_data(span_embeds),
        }
