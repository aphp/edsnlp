from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
)

import torch
from spacy.tokens import Doc, Span
from typing_extensions import Literal, TypedDict

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput
from edsnlp.pipelines.base import BaseComponent
from edsnlp.pipelines.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
    WordEmbeddingComponent,
)
from edsnlp.utils.filter import align_spans
from edsnlp.utils.span_getters import SpanGetterArg, get_spans

SpanPoolerBatchInput = TypedDict(
    "SpanPoolerBatchInput",
    {
        "embedding": BatchInput,
        "begins": torch.Tensor,
        "ends": torch.Tensor,
        "sequence_idx": torch.Tensor,
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


class SpanPooler(SpanEmbeddingComponent, BaseComponent):
    """
    The `eds.span_pooler` component is a trainable span embedding component. It
    generates span embeddings from a word embedding component and a span getter. It can
    be used to train a span classifier, as in `eds.span_classifier`.

    Parameters
    ----------
    nlp: PipelineProtocol
        Spacy vocabulary
    name: str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    span_getter: SpanGetterArg
        How to extract the candidate spans and the qualifiers
        to predict or train on.
    pooling_mode: Literal["max", "sum", "mean"]
        How word embeddings are aggregated into a single embedding per span.
    projection_mode: Literal["dot"]
        How embeddings converted into logits
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str,
        *,
        embedding: WordEmbeddingComponent,
        span_getter: SpanGetterArg,
        pooling_mode: Literal["max", "sum", "mean"] = "mean",
        projection_mode: Literal["dot"] = "dot",
    ):
        self.qualifiers = None
        self.output_size = embedding.output_size

        super().__init__(nlp, name)

        self.projection_mode = projection_mode
        self.pooling_mode = pooling_mode
        self.span_getter = span_getter

        if projection_mode != "dot":
            raise Exception(
                "Only scalar product is supported for label classification."
            )

        self.embedding = embedding

    def set_extensions(self):
        super().set_extensions()

        for qlf in self.qualifiers or ():
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        spans = list(get_spans(doc, self.span_getter))
        embedded_spans = list(get_spans(doc, self.embedding.span_getter))
        sequence_idx = []
        begins = []
        ends = []

        for i, (embedded_span, target_ents) in enumerate(
            zip(
                embedded_spans,
                align_spans(
                    source=spans,
                    target=embedded_spans,
                ),
            )
        ):
            start = embedded_span.start
            sequence_idx.extend([i] * len(spans))
            begins.extend([span.start - start for span in spans])
            ends.extend([span.end - start for span in spans])
        return {
            "embedding": self.embedding.preprocess(doc),
            "begins": begins,
            "ends": ends,
            "sequence_idx": sequence_idx,
            "$spans": spans,
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanPoolerBatchInput:
        collated: SpanPoolerBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "begins": torch.as_tensor([b for x in batch["begins"] for b in x]),
            "ends": torch.as_tensor([e for x in batch["ends"] for e in x]),
            "sequence_idx": torch.as_tensor(
                [e for x in batch["sequence_idx"] for e in x]
            ),
        }
        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanPoolerBatchInput) -> BatchOutput:
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
            return {
                "embeddings": torch.empty(0, self.output_size, device=device),
            }

        embeds = self.embedding.module_forward(batch["embedding"])["embeddings"]
        n_samples, n_words, dim = embeds.shape
        device = embeds.device

        flat_begins = n_words * batch["sequence_idx"] + batch["begins"]
        flat_ends = n_words * batch["sequence_idx"] + batch["ends"]
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

        return {
            "embeddings": span_embeds,
        }
