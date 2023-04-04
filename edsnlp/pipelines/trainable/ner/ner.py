from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from spacy.tokens import Doc, Span
from spacy.training import Example
from typing_extensions import NotRequired, TypedDict

from edsnlp.core.component import TorchComponent
from edsnlp.core.registry import registry
from edsnlp.utils.filter import filter_spans

from ..embeddings.typing import BatchInput, WordEmbeddingBatchOutput
from ..layers.crf import MultiLabelBIOULDecoder

NERBatchInput = TypedDict(
    "NERBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
    },
)
NERBatchOutput = TypedDict(
    "NERBatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "tags": Optional[torch.Tensor],
        "mask": Optional[torch.Tensor],
    },
)


class CRFMode(str, Enum):
    independent = "independent"
    joint = "joint"
    marginal = "marginal"


@registry.misc.register("span_getter")
def make_span_getter():
    def span_getter(doc):
        return doc.ents

    return span_getter


def nested_ner_exact_scorer(examples: Iterable[Example], **cfg) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`.

    Parameters
    ----------
    examples: Iterable[Example]
    cfg: Dict[str]
        - labels: Iterable[str] labels to take into account
        - spans_labels: Iterable[str] span group names to look into for entities

    Returns
    -------
    Dict[str, Any]
    """
    labels = set(cfg["labels"]) if "labels" in cfg is not None else None
    spans_labels = cfg.get("spans_labels", None)

    pred_spans = set()
    gold_spans = set()
    for eg_idx, eg in enumerate(examples):
        for span in (
            *eg.predicted.ents,
            *(
                span
                for name in (
                    spans_labels if spans_labels is not None else eg.reference.spans
                )
                for span in eg.predicted.spans.get(name, ())
            ),
        ):
            if labels is None or span.label_ in labels:
                pred_spans.add((eg_idx, span.start, span.end, span.label_))

        for span in (
            *eg.reference.ents,
            *(
                span
                for name in (
                    spans_labels if spans_labels is not None else eg.reference.spans
                )
                for span in eg.reference.spans.get(name, ())
            ),
        ):
            if labels is None or span.label_ in labels:
                gold_spans.add((eg_idx, span.start, span.end, span.label_))

    tp = len(pred_spans & gold_spans)

    return {
        "ents_p": tp / len(pred_spans) if pred_spans else float(tp == len(pred_spans)),
        "ents_r": tp / len(gold_spans) if gold_spans else float(tp == len(gold_spans)),
        "ents_f": 2 * tp / (len(pred_spans) + len(gold_spans))
        if pred_spans or gold_spans
        else float(len(pred_spans) == len(gold_spans)),
        "support": len(gold_spans),
    }


@registry.factory.register("eds.ner")
class TrainableNER(TorchComponent[NERBatchOutput, NERBatchInput]):
    def __init__(
        self,
        nlp,
        name: str,
        embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
        labels: List[str],
        span_getter: Callable[[Doc], Iterable[Span]],
        mode: CRFMode,
        scorer: Callable[[Iterable[Example]], Dict[str, Any]],
    ):
        super().__init__(nlp, name)
        self.name = name
        self.embedding = embedding
        self.span_getter = span_getter
        self.labels = list(labels)
        self.linear = torch.nn.Linear(
            self.embedding.output_size,
            len(labels) * 5,
        )
        self.crf = MultiLabelBIOULDecoder(
            1,
            with_start_end_transitions=True,
            learnable_transitions=False,
        )
        self.mode = mode
        self.scorer = scorer

    def score(self, examples: Sequence[Example]):
        return self.scorer(examples)

    def post_init(self, docs: Iterable[Doc]):
        # TODO, make span_getter default accessible from here
        labels = dict.fromkeys(self.labels)
        for doc in docs:
            for ent in self.span_getter(doc):
                labels[ent.label_] = None
        self.update_labels(list(labels.keys()))

    def update_labels(self, labels: Sequence[str]):
        n_old = len(self.labels)
        label_indices = dict(
            (
                *zip(self.labels, range(n_old)),
                *zip(labels, range(n_old, n_old + len(labels))),
            )
        )
        old_index = [label_indices[label] for label in self.labels]
        new_linear = torch.nn.Linear(
            self.embedding.output_size,
            len(labels) * 5,
        )
        new_linear.weight.data.view(-1, 5)[old_index] = self.linear.weight.data.view(
            -1, 5
        )
        new_linear.bias.data.view(-1, 5)[old_index] = self.linear.bias.data.view(-1, 5)
        self.linear = new_linear

        # Update initialization arguments
        self.labels = labels
        self.cfg["labels"] = labels

    def preprocess(self, doc):
        return {
            "embedding": self.embedding.preprocess(doc),
            "length": len(doc),
        }

    def preprocess_supervised(self, doc):
        targets = [[0] * len(self.labels) for _ in doc]
        for ent in self.span_getter(doc):
            label_idx = self.labels.index(ent.label_)
            if ent.start == ent.end - 1:
                targets[ent.start][label_idx] = 4
            else:
                targets[ent.start][label_idx] = 2
                targets[ent.end - 1][label_idx] = 3
                for i in range(ent.start + 1, ent.end - 1):
                    targets[i][label_idx] = 1
        return {
            **self.preprocess(doc),
            "targets": targets,
        }

    def collate(self, preps, device) -> NERBatchInput:
        collated: NERBatchInput = {
            "embedding": self.embedding.collate(preps["embedding"], device=device),
        }
        if "targets" in preps:
            max_len = max(map(len, preps["targets"]), default=0)
            targets = torch.as_tensor(
                [
                    row + [[-1] * len(self.labels)] * (max_len - len(row))
                    for row in preps["targets"]
                ],
                device=device,
                dtype=torch.long,
            )
            # targets = (targets.unsqueeze(-1) == torch.arange(5)).to(device)
            # mask = (targets[:, 0] != -1).to(device)
            collated["targets"] = targets
        return collated

    def forward(self, batch: NERBatchInput) -> NERBatchOutput:
        encoded = self.embedding(batch["embedding"])
        embeddings = encoded["embeddings"]
        mask = encoded["mask"]
        # batch words (labels tags) -> batch words labels tags
        scores = self.linear(embeddings).view((*embeddings.shape[:-1], -1, 5))
        loss = tags = None
        if "targets" in batch:
            if self.mode == "independent":
                loss = torch.nn.functional.cross_entropy(
                    scores.view(-1, 5),
                    batch["targets"].view(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
            elif self.mode == "joint":
                loss = self.crf(
                    scores,
                    mask,
                    batch["targets"].unsqueeze(-1) == torch.arange(5).to(scores.device),
                ).sum()
            elif self.mode == "marginal":
                loss = torch.nn.functional.cross_entropy(
                    self.crf.marginal(
                        scores,
                        mask,
                    ).view(-1, 5),
                    batch["targets"].view(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
        else:
            tags = self.crf.decode(scores, mask)
        return {
            "loss": loss,
            "tags": tags,
            "mask": mask,
        }

    def postprocess(self, docs: List[Doc], batch: NERBatchOutput):
        spans = self.crf.tags_to_spans(batch["tags"].cpu()).tolist()
        ents = [[] for _ in docs]
        span_groups = [{label: [] for label in self.labels} for _ in docs]
        for doc_idx, start, end, label_idx in spans:
            label = self.labels[label_idx]
            span = Span(docs[doc_idx], start, end, label)
            ents[doc_idx].append(span)
            span_groups[doc_idx][label].append(span)
        for doc, doc_ents, doc_span_groups in zip(docs, ents, span_groups):
            doc.ents = filter_spans(doc_ents)
            doc.spans.update(doc_span_groups)
        return docs

    def clean_gold_for_evaluation(self, gold: Doc) -> Doc:
        gold.ents = []
        gold.spans.clear()
        return gold
