from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from itertools import product
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import TypedDict

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.base import BaseRelationDetectorComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
    WordEmbeddingComponent,
)
from edsnlp.utils.span_getters import SpanGetterArg, get_spans
from edsnlp.utils.typing import AsList


def make_ranges(starts, ends):
    """
    Efficient computation and concat of ranges from starts and ends.

    Examples
    --------
    ```{ .python .no-check }

    starts = torch.tensor([0, 3, 6])
    ends = torch.tensor([2, 8, 8])
    make_ranges(starts, ends)
    #         <---> <----------->  <--->
    # tensor([0, 1, 3, 4, 5, 6, 7, 6, 7])
    ```

    Parameters
    ----------
    starts: torch.Tensor
    ends: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    assert starts.shape == ends.shape
    if 0 in ends.shape:
        return ends
    sizes = ends - starts
    offsets = sizes.cumsum(0)
    offsets = offsets.roll(1)
    res = torch.ones(offsets[0], dtype=torch.long)
    offsets[0] = 0
    res[offsets] = starts
    res[offsets[1:]] -= ends[:-1] - 1
    return res.cumsum(0)


logger = logging.getLogger(__name__)

FrameBatchInput = TypedDict(
    "FrameBatchInput",
    {
        "span_embedding": BatchInput,
        "word_embedding": BatchInput,
        "rel_head_idx": torch.Tensor,
        "rel_tail_idx": torch.Tensor,
        "rel_doc_idx": torch.Tensor,
        "rel_labels": torch.Tensor,
    },
)
"""
span_embedding: torch.FloatTensor
    Token embeddings to predict the tags from
"""


class RelationDetectorFFN(
    TorchComponent[BatchOutput, FrameBatchInput],
    BaseRelationDetectorComponent,
):
    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "rel_scope",
        *,
        span_embedding: SpanEmbeddingComponent,
        word_embedding: WordEmbeddingComponent,
        head_getter: SpanGetterArg,
        tail_getter: SpanGetterArg,
        labels: AsList[str],
        symmetric: bool = True,
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            head_getter=head_getter,
            tail_getter=tail_getter,
            labels=labels,
        )
        self.span_embedding = span_embedding
        self.word_embedding = word_embedding
        self.symmetric = symmetric
        # self.merge_mode = merge_mode

        hidden_size = (
            self.span_embedding.output_size * 2 + self.word_embedding.output_size
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # self.head_projection = torch.nn.Linear(hidden_size, hidden_size)
            # self.tail_projection = torch.nn.Linear(hidden_size, hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, len(self.labels))

    @property
    def span_getter(self):
        return self.embedding.span_getter

    def to_disk(self, path, *, exclude=set()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path, exclude=tuple()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        self.set_extensions()
        super().from_disk(path, exclude=exclude)

    def set_extensions(self):
        super().set_extensions()
        if not Span.has_extension("rel"):
            Span.set_extension("rel", default={})
        if not Span.has_extension("scope"):
            Span.set_extension("scope", default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc, supervised: int = False) -> Dict[str, Any]:
        rel_head_idx = []
        rel_tail_idx = []
        rel_labels = []

        all_spans = defaultdict(lambda: len(all_spans))
        head_spans = list(get_spans(doc, self.head_getter))
        tail_spans = list(get_spans(doc, self.tail_getter))

        for head, tail in product(head_spans, tail_spans):
            rel_head_idx.append(all_spans[head])
            rel_tail_idx.append(all_spans[tail])
            if supervised:
                rel_labels.append(
                    [
                        (
                            tail in head._.rel.get(lab, ())
                            or (self.symmetric and head in tail._.rel.get(lab, ()))
                        )
                        for lab in self.labels
                    ]
                )

        result = {
            "num_spans": len(all_spans),
            "rel_heads": rel_head_idx,
            "rel_tails": rel_tail_idx,
            "word_embedding": self.word_embedding.preprocess(doc, contexts=None),
            "span_embedding": self.span_embedding.preprocess(
                doc,
                spans=list(all_spans),
                contexts=None,
            ),
            "$spans": list(all_spans.keys()),
        }
        if supervised:
            result["rel_labels"] = rel_labels

        return result

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        return self.preprocess(doc, supervised=True)

    def collate(self, batch: Dict[str, Any]) -> FrameBatchInput:
        rel_heads = []
        rel_tails = []
        rel_doc_idx = []
        offset = 0
        for doc_idx, feats in enumerate(
            zip(
                batch["rel_heads"],
                batch["rel_tails"],
                batch["num_spans"],
            )
        ):
            doc_rel_heads, doc_rel_tails, doc_num_spans = feats
            rel_heads.extend([x + offset for x in doc_rel_heads])
            rel_tails.extend([x + offset for x in doc_rel_tails])
            rel_doc_idx.extend([doc_idx] * len(doc_rel_heads))
            offset += batch["num_spans"][doc_idx]

        collated: FrameBatchInput = {
            "rel_head_idx": torch.as_tensor(rel_heads, dtype=torch.long),
            "rel_tail_idx": torch.as_tensor(rel_tails, dtype=torch.long),
            "rel_doc_idx": torch.as_tensor(rel_doc_idx, dtype=torch.long),
            "span_embedding": self.span_embedding.collate(batch["span_embedding"]),
            "word_embedding": self.word_embedding.collate(batch["word_embedding"]),
        }

        if "rel_labels" in batch:
            collated["rel_labels"] = torch.as_tensor(
                [labs for doc_labels in batch["rel_labels"] for labs in doc_labels]
            )
        return collated

    def compute_inter_span_embeds(self, word_embeds, begins, ends, head_idx, tail_idx):
        _, n_words, dim = word_embeds.shape
        if 0 in begins.shape or 0 in head_idx.shape:
            return torch.zeros(
                0, dim, dtype=word_embeds.dtype, device=word_embeds.device
            )

        flat_begins = torch.minimum(
            ends[head_idx],
            ends[tail_idx],
        )
        flat_ends = torch.maximum(
            begins[head_idx],
            begins[tail_idx],
        )
        flat_embeds = word_embeds.view(-1, dim)
        flat_indices = make_ranges(flat_begins, flat_ends)
        flat_offsets = (flat_ends - flat_begins).cumsum(0).roll(1)
        flat_offsets[0] = 0
        inter_span_embeds = torch.nn.functional.embedding_bag(  # type: ignore
            input=flat_indices,
            weight=flat_embeds,
            offsets=flat_offsets,
            mode="mean",
        )
        return inter_span_embeds

    # noinspection SpellCheckingInspection
    def forward(self, batch: FrameBatchInput) -> BatchOutput:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans

        Parameters
        ----------
        batch: SpanQualifierBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        word_embeds = self.word_embedding(batch["word_embedding"])["embeddings"]
        span_embeds = self.span_embedding(batch["span_embedding"])["embeddings"]

        inter_span_embeds = self.compute_inter_span_embeds(
            word_embeds=word_embeds,
            begins=batch["span_embedding"]["begins"],
            ends=batch["span_embedding"]["ends"],
            head_idx=batch["rel_head_idx"],
            tail_idx=batch["rel_tail_idx"],
        )
        rel_embeds = torch.cat(
            [
                span_embeds[batch["rel_head_idx"]],
                inter_span_embeds,
                span_embeds[batch["rel_tail_idx"]],
            ],
            dim=-1,
        )
        logits = self.classifier(rel_embeds)

        losses = pred = None
        if "rel_labels" in batch:
            losses = []
            target = batch["rel_labels"].float()
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
        else:
            pred = logits > 0

        return {
            "loss": sum(losses) if losses is not None else None,
            "pred": pred,
        }

    def postprocess(
        self,
        docs: List[Doc],
        results: BatchOutput,
        inputs: List[Dict[str, Any]],
    ):
        """
        Extract predicted relations from forward's "pred" field (boolean tensor)
        and annotated them on the head._.rel attribute (dictionary)
        Parameters
        ----------
        docs: Sequence[Doc]
            List of documents to update
        results: BatchOutput
            Batch of predictions, as returned by the forward method
        inputs: BatchInput
            List of preprocessed features, as returned by the preprocess method

        Returns
        -------
        """
        all_heads = [
            prep["$spans"][idx] for prep in inputs for idx in prep["rel_heads"]
        ]
        all_tails = [
            prep["$spans"][idx] for prep in inputs for idx in prep["rel_tails"]
        ]
        for pair_idx, label_idx in results["pred"].nonzero(as_tuple=False).tolist():
            head = all_heads[pair_idx]
            tail = all_tails[pair_idx]
            label = self.labels[label_idx]
            head._.rel.setdefault(label, set()).add(tail)
            if self.symmetric:
                tail._.rel.setdefault(label, set()).add(head)
        return docs