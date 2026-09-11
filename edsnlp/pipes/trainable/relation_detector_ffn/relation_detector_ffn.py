from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import NotRequired, TypedDict

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.base import BaseRelationDetectorComponent
from edsnlp.pipes.trainable.embeddings.typing import SpanEmbeddingComponent
from edsnlp.utils.span_getters import RelationCandidateGetter, get_spans
from edsnlp.utils.typing import AsList

RelationBatchInput = TypedDict(
    "RelationBatchInput",
    {
        "span_embedding": BatchInput,
        "inter_embedding": Optional[BatchInput],
        "rel_head_idx": torch.Tensor,
        "rel_tail_idx": torch.Tensor,
        "rel_labels": NotRequired[torch.Tensor],
        "stats": Dict[str, Any],
    },
)
"""
Attributes
----------
span_embedding: BatchInput
    Output of the span embedding component's collate method
inter_embedding: Optional[BatchInput]
    Collated inter-span embeddings, or None when the component is disabled
rel_head_idx: torch.Tensor
    Indices of head spans for each candidate pair
rel_tail_idx: torch.Tensor
    Indices of tail spans for each candidate pair
rel_labels: torch.Tensor, optional
    Gold labels for each candidate pair, shape (num_candidates, num_labels)
stats: Dict[str, Any]
    Statistics about the batch, ie. "relation_candidates": int
"""


class MLP(torch.nn.Module):
    """
    Project candidate pair embeddings into features for the relation classifier
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.0
    ):
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        return self.output(F.gelu(self.hidden(self.dropout(x))))


class RelationDetectorFFN(
    TorchComponent[BatchOutput, RelationBatchInput],
    BaseRelationDetectorComponent,
):
    """
    The `eds.relation_detector_ffn` component is a trainable relation detector that
    predicts relations between pairs of spans. For each candidate pair, it computes
    a score for every relation label.

    How it works
    ------------
    The component builds relation candidates from a user-provided configuration
    (`candidate_getter`) that specifies which spans can be heads and tails. For each
    candidate pair, it computes an embedding by concatenating:

    1. the embedding of the head span
    2. the embedding of the words that lie between the head and the tail (if
       `inter_span_embedding` is provided)
    3. the embedding of the tail span

    This concatenated vector is passed through a small feed-forward network followed by
    a linear classifier to produce one logit per relation label. Training uses a
    binary cross-entropy loss for multi-label classification. At inference time,
    predictions are obtained by applying a zero threshold to the logits (i.e.
    probability > 0.5).

    The predicted relations are written to `span._.rel` as a dictionary that maps each
    label to the set of related spans. If a candidate configuration is marked as
    symmetric, the relation is added in both directions.

    Example
    -------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.relation_detector_ffn(
            # Span representations for head/tail and for the words between them
            span_embedding=eds.span_pooler(
                pooling_mode="mean",
                embedding=eds.transformer(
                    model="hf-internal-testing/tiny-random-bert",
                    window=128,
                    stride=96,
                ),
            ),
            # Provide `inter_span_embedding` to include between-words embeddings
            inter_span_embedding=eds.span_pooler(  # (1)!
                pooling_mode="mean",
                embedding=eds.transformer(
                    model="hf-internal-testing/tiny-random-bert",
                    window=128,
                    stride=96,
                ),
            ),
            # Build relation candidates between drugs and problems
            candidate_getter=[
                {
                    "head": {"ents": ["drug"]},
                    "tail": {"ents": ["problem"]},
                    "labels": ["treats"],
                    "symmetric": False,
                }
            ],
            hidden_size=128,
            dropout_p=0.1,
        ),
        name="rel_ffn",
    )
    ```
    1. You can either provide no `inter_span_embedding` (in which case only head and
       tail embeddings are used), or reuse the same component as `span_embedding` or
       use a different one like in this example.

    To train the model, provide supervision by filling, for each head span, the
    extension `span._.rel` as a mapping from label to a set of tail spans. For
    symmetric relations, you may annotate only one side when `symmetric=True` in the
    candidate configuration. We should have
    span._.rel = {"RELATION_LABEL": [related_span1, related_span2, ...]}

    For example, to indicate that a drug (span1) treats a problem (span2), you would do:
    `span1._.rel = {"treats": (span2,)}`


    Extensions
    ----------
    The component declares the following `Span` extensions:

    - `span._.rel`: dict that maps a relation label to a set of related spans

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    span_embedding : SpanEmbeddingComponent
        Embedding used to represent head and tail spans
    inter_span_embedding : Optional[SpanEmbeddingComponent]
        Embedding used to represent the words between the head and tail. When
        provided, its embedding is concatenated between the head and tail embeddings.
        If `None`, only head and tail embeddings are used.
    candidate_getter : AsList[RelationCandidateGetter]
        Configuration that defines candidate pairs and their possible labels.
        Each entry is a dict with keys `head`, `tail`, `labels`, and optional
        `label_filter` and `symmetric`.
    hidden_size : int
        Hidden size of the feed-forward network
    dropout_p : float
        Dropout probability applied before the hidden layer
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "relation_detector_ffn",
        *,
        span_embedding: SpanEmbeddingComponent,
        inter_span_embedding: Optional[SpanEmbeddingComponent] = None,
        candidate_getter: AsList[RelationCandidateGetter],
        hidden_size: int = 128,
        dropout_p: float = 0.0,
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            candidate_getter=candidate_getter,
        )
        self.span_embedding = span_embedding
        self.inter_span_embedding = inter_span_embedding

        embed_size = self.span_embedding.output_size * 2 + (
            self.inter_span_embedding.output_size
            if self.inter_span_embedding is not None
            else 0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.mlp = MLP(embed_size, hidden_size, hidden_size, dropout_p)
            self.classifier = torch.nn.Linear(hidden_size, len(self.labels))

    def set_extensions(self):
        super().set_extensions()
        if not Span.has_extension("rel"):
            Span.set_extension("rel", default={})

    def preprocess(self, doc: Doc, supervised: bool = False) -> Dict[str, Any]:
        """
        Build candidate pairs and span embeddings with optional relation targets
        """
        rel_head_idx = []
        rel_tail_idx = []
        rel_labels = []
        rel_getter_indices = []
        inter_spans = []

        all_spans = defaultdict(lambda: len(all_spans))
        seen = set()

        for getter_idx, getter in enumerate(self.candidate_getter):
            head_spans = list(get_spans(doc, getter["head"]))
            tail_spans = list(get_spans(doc, getter["tail"]))
            lab_filter = getter["label_filter"]
            for head, tail in product(head_spans, tail_spans):
                if (
                    lab_filter
                    and head.label_ in lab_filter
                    and tail.label_ not in lab_filter[head.label_]
                ):
                    continue
                # The first matching getter owns each candidate pair
                if (head, tail) in seen:
                    continue
                seen.add((head, tail))
                rel_head_idx.append(all_spans[head])
                rel_tail_idx.append(all_spans[tail])
                inter_beg = min(head.end, tail.end)
                inter_end = max(head.start, tail.start)
                inter_spans.append(doc[inter_beg:inter_end])
                rel_getter_indices.append(getter_idx)
                if supervised:
                    rel_labels.append(
                        [
                            (
                                tail in head._.rel.get(lab, ())
                                or (
                                    getter["symmetric"]
                                    and head in tail._.rel.get(lab, ())
                                )
                            )
                            for lab in self.labels
                        ]
                    )

        result = {
            "num_spans": len(all_spans),
            "rel_heads": rel_head_idx,
            "rel_tails": rel_tail_idx,
            "inter_embedding": self.inter_span_embedding.preprocess(
                doc,
                spans=inter_spans,
                contexts=None,
            )
            if self.inter_span_embedding is not None
            else None,
            "span_embedding": self.span_embedding.preprocess(
                doc,
                spans=list(all_spans),
                contexts=None,
            ),
            "$spans": list(all_spans.keys()),
            "$getter": rel_getter_indices,
            "stats": {
                "relation_candidates": len(rel_head_idx),
            },
        }
        if supervised:
            result["rel_labels"] = rel_labels

        return result

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        return self.preprocess(doc, supervised=True)

    def collate(self, batch: Dict[str, Any]) -> RelationBatchInput:
        """
        Offset candidate indices into the batch of head and tail span embeddings
        """
        rel_heads = []
        rel_tails = []
        offset = 0
        for doc_rel_heads, doc_rel_tails, doc_num_spans in zip(
            batch["rel_heads"], batch["rel_tails"], batch["num_spans"]
        ):
            rel_heads.extend([x + offset for x in doc_rel_heads])
            rel_tails.extend([x + offset for x in doc_rel_tails])
            offset += doc_num_spans

        collated: RelationBatchInput = {
            "rel_head_idx": torch.as_tensor(rel_heads, dtype=torch.long),
            "rel_tail_idx": torch.as_tensor(rel_tails, dtype=torch.long),
            "span_embedding": self.span_embedding.collate(batch["span_embedding"]),
            "inter_embedding": self.inter_span_embedding.collate(
                batch["inter_embedding"]
            )
            if self.inter_span_embedding is not None
            else None,
            "stats": {"relation_candidates": len(rel_heads)},
        }

        if "rel_labels" in batch:
            collated["rel_labels"] = torch.as_tensor(
                [labs for doc_labels in batch["rel_labels"] for labs in doc_labels]
            ).view(-1, self.classifier.out_features)
        return collated

    def forward(self, batch: RelationBatchInput) -> BatchOutput:
        """
        Compute relation loss or predictions from candidate pair embeddings

        Parameters
        ----------
        batch: RelationBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        span_embeds = self.span_embedding(batch["span_embedding"])["embeddings"]
        if self.inter_span_embedding is not None:
            inter_embeds = self.inter_span_embedding(batch["inter_embedding"])[
                "embeddings"
            ]
            rel_embeds = torch.cat(
                [
                    span_embeds[batch["rel_head_idx"]],
                    inter_embeds,
                    span_embeds[batch["rel_tail_idx"]],
                ],
                dim=-1,
            )
        else:
            rel_embeds = torch.cat(
                [
                    span_embeds[batch["rel_head_idx"]],
                    span_embeds[batch["rel_tail_idx"]],
                ],
                dim=-1,
            )
        rel_embeds = self.mlp(rel_embeds)
        logits = self.classifier(rel_embeds)

        loss = pred = None
        if "rel_labels" in batch:
            # Documents without candidate pairs contribute zero loss
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["rel_labels"].float(), reduction="sum"
            ) / max(1, batch["stats"]["relation_candidates"])
        else:
            pred = logits > 0

        return {
            "loss": loss,
            "pred": pred,
        }

    def postprocess(
        self,
        docs: List[Doc],
        results: BatchOutput,
        inputs: List[Dict[str, Any]],
    ):
        """
        Store predicted tail spans in each head span relation dictionary

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
        List[Doc]
            Documents with predicted relations
        """
        all_heads = [p["$spans"][idx] for p in inputs for idx in p["rel_heads"]]
        all_tails = [p["$spans"][idx] for p in inputs for idx in p["rel_tails"]]
        getter_indices = [idx for p in inputs for idx in p["$getter"]]
        for pair_idx, label_idx in results["pred"].nonzero(as_tuple=False).tolist():
            head = all_heads[pair_idx]
            tail = all_tails[pair_idx]
            label = self.labels[label_idx]
            related_spans = head._.rel.setdefault(label, set())
            if not isinstance(related_spans, set):
                related_spans = head._.rel[label] = set(related_spans)
            related_spans.add(tail)
            if self.candidate_getter[getter_indices[pair_idx]]["symmetric"]:
                related_spans = tail._.rel.setdefault(label, set())
                if not isinstance(related_spans, set):
                    related_spans = tail._.rel[label] = set(related_spans)
                related_spans.add(head)
        return docs
