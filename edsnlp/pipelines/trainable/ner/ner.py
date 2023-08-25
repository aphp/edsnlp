from __future__ import annotations

import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from pydantic import StrictStr
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp import Pipeline
from edsnlp.core.component import TorchComponent
from edsnlp.utils.filter import filter_spans
from edsnlp.pipelines.base import (
    SpanGetter,
    SpanGetterMapping,
    get_spans,
)

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
    },
)


class CRFMode(str, Enum):
    independent = "independent"
    joint = "joint"
    marginal = "marginal"

class TrainableNER(TorchComponent[NERBatchOutput, NERBatchInput]):
    def __init__(
        self,
        nlp: Pipeline = None,
        name: Optional[str] = None,
        *,
        embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
        target_span_getter: SpanGetter,
        labels: Optional[List[str]] = None,
        to_ents: Union[bool, List[str]] = None,
        to_span_groups: Union[StrictStr, Dict[str, Union[bool, List[str]]]] = None,
        mode: Literal["independent", "joint", "marginal"],
    ):
        super().__init__(nlp, name)

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
        self.to_ents = to_ents
        self.to_span_groups: Dict[str, Union[bool, List[str]]] = (
            {to_span_groups: True}
            if isinstance(to_span_groups, str)
            else to_span_groups
        )
        if callable(target_span_getter) and (
            self.to_ents is None or self.to_span_groups is None
        ):
            raise ValueError(
                "If `target_span_getter` is callable, `to_ents` or `to_span_groups` "
                "cannot be inferred and must both be set manually"
            )
        if (
            isinstance(target_span_getter, dict) and "labels" in target_span_getter
        ) and self.labels is not None:
            raise ValueError(
                "You cannot set both the `labels` key of the `target_span_getter` "
                "parameter and the `labels` parameter."
            )

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
        if (
            self.labels is not None
            and self.to_ents is not None
            and self.to_span_groups is not None
        ):
            return

        inferred_labels = set()

        to_ents = []
        to_span_groups = defaultdict(lambda: [])

        for doc in docs:
            if callable(self.target_span_getter):
                for ent in self.target_span_getter(doc):
                    inferred_labels.add(ent.label_)
            else:
                if "span_groups" in self.target_span_getter:
                    for group in self.target_span_getter["span_groups"]:
                        for span in doc.spans.get(group, ()):
                            if self.labels is None or span.label_ in self.labels:
                                inferred_labels.add(span.label_)
                                to_span_groups[group].append(span.label_)
                elif "ents" in self.target_span_getter:
                    for span in doc.ents:
                        if self.labels is None or span.label_ in self.labels:
                            inferred_labels.add(span.label_)
                            to_ents.append(span.label_)
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

        if not self.labels:
            raise ValueError(
                "No labels were inferred from the data. Please check your data and "
                "the `target_span_getter` parameter."
            )

        if self.to_ents is None:
            self.to_ents = to_ents
            self.cfg["to_ents"] = self.to_ents
        if self.to_span_groups is None:
            self.to_span_groups = dict(to_span_groups)
            self.cfg["to_span_groups"] = self.to_span_groups

    def update_labels(self, labels: Sequence[str]):
        original_labels = self.labels if self.labels is not None else ()
        n_old = len(original_labels)
        label_indices = dict(
            (
                *zip(original_labels, range(n_old)),
                *zip(labels, range(n_old, n_old + len(labels))),
            )
        )
        old_index = [label_indices[label] for label in original_labels]
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

    def get_target_spans(self, doc):
        return (
            self.target_span_getter(doc)
            if callable(self.target_span_getter)
            else get_spans(doc, self.target_span_getter)
        )

    def postprocess(self, docs: List[Doc], batch: NERBatchOutput):
        spans = self.crf.tags_to_spans(batch["tags"].cpu()).tolist()
        ents = [[] for _ in docs]
        if self.to_span_groups is None or self.to_ents is None:
            raise ValueError(
                f"The {self.__class__.__name__} component still has to infer the "
                f"`to_ents` and `to_span_groups` parameters. Please call "
                f"`nlp.post_init(...)` before running it on some new data, or set "
                f"both parameters manually."
            )
        span_groups = [{label: [] for label in self.to_span_groups} for _ in docs]
        for doc_idx, start, end, label_idx in spans:
            label = self.labels[label_idx]
            span = Span(docs[doc_idx], start, end, label)
            if self.to_ents is True or label in self.to_ents:
                ents[doc_idx].append(span)
            for group_name, group_spans in span_groups[doc_idx].items():
                if (
                    self.to_span_groups[group_name] is True
                    or label in self.to_span_groups[group_name]
                ):
                    span_groups[doc_idx][group_name].append(span)
        for doc, doc_ents, doc_span_groups in zip(docs, ents, span_groups):
            if doc_ents:
                doc.ents = filter_spans((*doc.ents, *doc_ents))
            if self.to_span_groups:
                doc.spans.update(doc_span_groups)
        return docs
