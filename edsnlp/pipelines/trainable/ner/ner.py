from __future__ import annotations

import warnings
from collections import defaultdict
from enum import Enum
from typing import Callable, Iterable, List, Optional, Sequence, Union

import torch
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp import Pipeline
from edsnlp.core.component import TorchComponent
from edsnlp.pipelines.base import (
    BaseNERComponent,
    SpanGetterArg,
    SpanGetterMapping,
    SpanSetterArg,
    get_spans,
)
from edsnlp.pipelines.trainable.embeddings.typing import (
    BatchInput,
    WordEmbeddingBatchOutput,
)
from edsnlp.pipelines.trainable.layers.crf import MultiLabelBIOULDecoder

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
    },
)


class CRFMode(str, Enum):
    independent = "independent"
    joint = "joint"
    marginal = "marginal"


class TrainableNER(TorchComponent[NERBatchOutput, NERBatchInput], BaseNERComponent):
    """
    Initialize a general named entity recognizer (with or without nested or
    overlapping entities).

    Parameters
    ----------
    nlp: PipelineProtocol
        The current nlp object
    name: str
        Name of the component
    embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput]
        The word embedding component
    target_span_getter: Callable[[Doc], Iterable[Span]]
        Method to call to get the gold spans from a document, for scoring or training.
        By default, takes all entities in `doc.ents`, but we recommend you specify
        a given span group name instead.
    labels: List[str]
        The labels to predict. The labels can also be inferred from the data
        during `nlp.post_init(...)`
    span_setter: Optional[SpanSetterArg]
        The span setter to use to set the predicted spans on the Doc object. If None,
        the component will infer the span setter from the target_span_getter config.
    infer_span_setter: Optional[bool]
        Whether to complete the span setter from the target_span_getter config.
        False by default, unless the span_setter is None.
    mode: Literal["independent", "joint", "marginal"]
        The CRF mode to use: independent, joint or marginal
    """

    def __init__(
        self,
        nlp: Pipeline = None,
        name: Optional[str] = "eds.ner",
        *,
        embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
        target_span_getter: SpanGetterArg,
        labels: Optional[List[str]] = None,
        span_setter: Optional[SpanSetterArg] = None,
        infer_span_setter: Optional[bool] = None,
        mode: Literal["independent", "joint", "marginal"],
    ):
        if (
            isinstance(target_span_getter, dict) and "labels" in target_span_getter
        ) and labels is not None:
            raise ValueError(
                "You cannot set both the `labels` key of the `target_span_getter` "
                "parameter and the `labels` parameter."
            )

        super().__init__(
            nlp=nlp,
            name=name,
            span_setter=span_setter or {},
        )
        self.infer_span_setter = (
            span_setter is None if infer_span_setter is None else infer_span_setter
        )

        self.embedding = embedding
        self.labels = labels
        self.linear = torch.nn.Linear(
            self.embedding.output_size,
            0 if labels is None else (len(labels) * 5),
        )
        self.crf = MultiLabelBIOULDecoder(
            1,
            with_start_end_transitions=True,
            learnable_transitions=False,
        )
        self.mode = mode

        if isinstance(target_span_getter, list):
            target_span_getter = {"span_groups": target_span_getter}

        self.target_span_getter: Union[
            SpanGetterMapping,
            Callable[[Doc], Iterable[Span]],
        ] = target_span_getter

    def post_init(self, docs: Iterable[Doc]):
        """
        Update the labels based on the data and the span getter,
        and fills in the to_ents and to_span_groups if necessary

        Parameters
        ----------
        docs

        Returns
        -------

        """
        if self.labels is not None and not self.infer_span_setter:
            return

        inferred_labels = set()

        span_setter = defaultdict(lambda: [])

        for doc in docs:
            if callable(self.target_span_getter):
                for ent in self.target_span_getter(doc):
                    inferred_labels.add(ent.label_)
            else:
                for key, span_filter in self.target_span_getter.items():
                    candidates = doc.spans.get(key, ()) if key != "ents" else doc.ents
                    if span_filter is True:
                        filtered_candidates = candidates
                    else:
                        filtered_candidates = [
                            span
                            for span in candidates
                            if span.label_ in span_filter
                            and (self.labels is None or span.label_ in self.labels)
                        ]
                    for span in filtered_candidates:
                        inferred_labels.add(span.label_)
                        span_setter[key].append(span.label_)
        if self.labels is not None:
            assert inferred_labels <= set(self.labels), (
                "Some inferred labels are not present in the labels "
                f"passed to the component: {inferred_labels - set(self.labels)}"
            )
            if inferred_labels < set(self.labels):
                warnings.warn(
                    "Some labels passed to the trainable NER component are not "
                    "present in the inferred labels list: "
                    f"{set(self.labels) - inferred_labels}"
                )
        else:
            self.update_labels(sorted(inferred_labels))

        self.span_setter = {
            **self.span_setter,
            **{
                key: value
                for key, value in span_setter.items()
                if key not in self.span_setter
            },
        }

        if not self.labels:
            raise ValueError(
                "No labels were inferred from the data. Please check your data and "
                "the `target_span_getter` parameter."
            )

    def update_labels(self, labels: Sequence[str]):
        old_labels = self.labels if self.labels is not None else ()
        n_old = len(old_labels)
        dict(
            reversed(
                (
                    *zip(old_labels, range(n_old)),
                    *zip(labels, range(n_old, n_old + len(labels))),
                )
            )
        )
        old_index = (
            torch.arange(len(old_labels) * 5)
            .view(-1, 5)[
                [labels.index(label) for label in old_labels if label in labels]
            ]
            .view(-1)
        )
        new_index = (
            torch.arange(len(labels) * 5)
            .view(-1, 5)[
                [old_labels.index(label) for label in old_labels if label in labels]
            ]
            .view(-1)
        )
        new_linear = torch.nn.Linear(self.embedding.output_size, len(labels) * 5)
        new_linear.weight.data[new_index] = self.linear.weight.data[old_index]
        new_linear.bias.data[new_index] = self.linear.bias.data[old_index]
        self.linear.weight = new_linear.weight
        self.linear.bias = new_linear.bias

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
        for ent in self.get_target_spans(doc):
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
            tags = self.crf.decode(
                scores, mask
            )  # tags = scores.argmax(-1).masked_fill(~mask.unsqueeze(-1), 0)
        return {
            "loss": loss,
            "tags": tags,
        }

    def get_target_spans(self, doc) -> Iterable[Span]:
        return (
            self.target_span_getter(doc)
            if callable(self.target_span_getter)
            else get_spans(doc, self.target_span_getter)
        )

    def postprocess(self, docs: List[Doc], batch: NERBatchOutput):
        spans: List[List[Span]] = [[] for _ in docs]
        for doc_idx, start, end, label_idx in self.crf.tags_to_spans(
            batch["tags"].cpu()
        ).tolist():
            spans[doc_idx].append(
                Span(
                    docs[doc_idx],
                    start,
                    end,
                    self.labels[label_idx],
                )
            )
        for doc, doc_spans in zip(docs, spans):
            self.set_spans(doc, doc_spans)
        return docs
