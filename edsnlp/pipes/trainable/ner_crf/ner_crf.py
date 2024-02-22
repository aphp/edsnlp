from __future__ import annotations

import warnings
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp import Pipeline
from edsnlp.core.torch_component import TorchComponent
from edsnlp.pipes.base import BaseNERComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    BatchInput,
    WordEmbeddingComponent,
)
from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder
from edsnlp.utils.filter import align_spans, filter_spans
from edsnlp.utils.span_getters import (
    SpanGetterArg,
    SpanGetterMapping,
    SpanSetter,
    SpanSetterArg,
    get_spans,
)
from edsnlp.utils.torch import make_windows

NERBatchInput = TypedDict(
    "NERBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
        "window_indices": NotRequired[torch.Tensor],
        "window_indexer": NotRequired[torch.Tensor],
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


class TrainableNerCrf(TorchComponent[NERBatchOutput, NERBatchInput], BaseNERComponent):
    """
    The `eds.ner_crf` component is a general purpose trainable named entity recognizer.
    It can extract:

    - flat entities
    - overlapping entities of different labels

    However, at the moment, the model cannot currently extract entities that are
    nested inside larger entities of the same label.

    It is based on a CRF (Conditional Random Field) layer and should therefore work
    well on dataset composed of entities will ill-defined boundaries. We offer a
    compromise between speed and performance by allowing the user to specify a window
    size for the CRF layer. The smaller the window, the faster the model will be, but
    at the cost of degraded performance.

    The pipeline assigns both `doc.ents` (in which overlapping entities are filtered
    out) and `doc.spans`. These destinations can be inferred from the
    `target_span_getter` parameter, combined with the `post_init` step.

    Architecture
    ------------
    The model performs token classification using the BIOUL (Begin, Inside, Outside,
    Unary, Last) tagging scheme. To extract overlapping entities, each label has its
    own tag sequence, so the model predicts `n_labels` sequences of O, I, B, L, U tags.
    The architecture is displayed in the figure below.

    To enforce the tagging scheme, (ex: I cannot follow O but only B, ...), we use a
    stack of CRF (Conditional Random Fields) layers, one per label during both training
    and prediction.

    <figure markdown>
      ![Nested NER architecture](/assets/images/edsnlp-ner.svg)
      <figcaption>Nested NER architecture</figcaption>
    </figure>

    Examples
    --------
    Let us define a pipeline composed of a transformer, and a NER component.

    ```{ .python }
    from pathlib import Path

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config=dict(
            embedding=nlp.get_pipe("transformer"),
            mode="joint",
            target_span_getter=["ents", "ner-preds"],
            window=10,
        )
    )
    ```

    To train the model, refer to the [Training](/tutorials/training.md) tutorial.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    target_span_getter : SpanGetterArg
        Method to call to get the gold spans from a document, for scoring or training.
        By default, takes all entities in `doc.ents`, but we recommend you specify
        a given span group name instead.
    labels : List[str]
        The labels to predict. The labels can also be inferred from the data
        during `nlp.post_init(...)`
    span_setter : Optional[SpanSetterArg]
        The span setter to use to set the predicted spans on the Doc object. If None,
        the component will infer the span setter from the target_span_getter config.
    infer_span_setter : Optional[bool]
        Whether to complete the span setter from the target_span_getter config.
        False by default, unless the span_setter is None.
    mode : Literal["independent", "joint", "marginal"]
        The CRF mode to use : independent, joint or marginal
    window : int
        The window size to use for the CRF. If 0, will use the whole document, at
        the cost of a longer computation time. If 1, this is equivalent to assuming
        that the tags are independent and will the component be faster, but with
        degraded performance. Empirically, we found that a window size of 10 or 20
        works well.
    stride : Optional[int]
        The stride to use for the CRF windows. Defaults to `window - 2`.

    Authors and citation
    --------------------
    The `eds.ner_crf` pipeline was developed by AP-HP's Data Science team.

    The deep learning model was adapted from [@wajsburt:tel-03624928].
    """

    span_setter: SpanSetter

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: Optional[str] = "eds.ner_crf",
        *,
        embedding: WordEmbeddingComponent,
        target_span_getter: SpanGetterArg = {"ents": True},
        labels: Optional[List[str]] = None,
        span_setter: Optional[SpanSetterArg] = None,
        infer_span_setter: Optional[bool] = None,
        mode: Literal["independent", "joint", "marginal"],
        window: int = 40,
        stride: Optional[int] = None,
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
            with_start_end_transitions=window < 1,
            learnable_transitions=False,
        )
        self.mode = mode
        if stride is None:
            stride = window // 2
        stride = stride if window > 0 else 0
        if window < stride:
            raise ValueError(
                "The window size must be greater than or equal to the stride."
            )
        if window == 1 and self.mode != "independent":
            warnings.warn(
                "The TrainableNerCrf module will be using a window size equals to 1"
                "(i.e. assumes tags are independent) while trained in non "
                "`independent` mode. This may lead to degraded performance."
            )
        self.window: int = window
        self.stride: int = stride

        self.target_span_getter: Union[
            SpanGetterMapping,
            Callable[[Doc], Iterable[Span]],
        ] = target_span_getter

    def post_init(self, docs: Iterable[Doc], exclude: Set[str]):
        """
        Update the labels based on the data and the span getter,
        and fills in the to_ents and to_span_groups if necessary

        Parameters
        ----------
        docs: Iterable[Doc]
            The documents to use to infer the labels
        exclude: Set[str]
            Components to exclude from the post initialization
        """
        super().post_init(docs, exclude)

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
                    filtered_candidates = (
                        candidates
                        if span_filter is True
                        else [
                            span
                            for span in candidates
                            if span.label_ in span_filter
                            and (self.labels is None or span.label_ in self.labels)
                        ]
                    )
                    for span in filtered_candidates:
                        inferred_labels.add(span.label_)
                        span_setter[key].append(span.label_)
        if self.labels is not None:
            if inferred_labels != set(self.labels):
                warnings.warn(
                    "The labels inferred from the data are different from the "
                    "labels passed to the component. Differing labels are "
                    f"{sorted(set(self.labels) ^ inferred_labels)}",
                    UserWarning,
                )
        else:
            self.update_labels(sorted(inferred_labels))

        self.span_setter = (
            {
                **self.span_setter,
                **{
                    key: sorted(set(value))
                    for key, value in span_setter.items()
                    if key not in self.span_setter
                },
            }
            if self.infer_span_setter
            else self.span_setter
        )

        if not self.labels:
            raise ValueError(
                "No labels were inferred from the data. Please check your data and "
                f"the `target_span_getter` parameter ({self.target_span_getter})."
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

    @property
    def cfg(self):
        return {
            "labels": self.labels,
            "span_setter": self.span_setter,
            "infer_span_setter": self.infer_span_setter,
            "mode": self.mode,
            "window": self.window,
            "stride": self.stride,
        }

    def preprocess(self, doc):
        if self.labels is None:
            raise ValueError(
                "The component was not initialized with any labels. Please "
                "initialize it with the `labels` parameter, pass a list of "
                "labels to the `update_labels` method, or call the `post_init` "
                "method with a list of documents."
            )
        return {
            "embedding": self.embedding.preprocess(doc),
            "length": len(doc),
        }

    def preprocess_supervised(self, doc):
        embedded_spans = list(get_spans(doc, self.embedding.span_getter))
        tags = []

        discarded = []
        for embedded_span, target_ents in zip(
            embedded_spans,
            align_spans(
                source=list(self.get_target_spans(doc)),
                target=embedded_spans,
            ),
        ):
            span_tags = [[0] * len(self.labels) for _ in range(len(embedded_span))]
            start = embedded_span.start
            by_label = defaultdict(list)
            for ent in target_ents:
                by_label[ent.label_].append(ent)
            filtered = []
            for label, spans in by_label.items():
                filtered[len(filtered) :], discarded[len(discarded) :] = filter_spans(
                    by_label[label], return_discarded=True
                )

            for ent in filtered:
                label_idx = self.labels.index(ent.label_)
                if ent.start == ent.end - 1:
                    span_tags[ent.start - start][label_idx] = 4
                else:
                    span_tags[ent.start - start][label_idx] = 2
                    span_tags[ent.end - 1 - start][label_idx] = 3
                    for i in range(ent.start + 1 - start, ent.end - 1 - start):
                        span_tags[i][label_idx] = 1
            tags.append(span_tags)

        if discarded:
            warnings.warn(
                f"Some spans in were discarded {doc._.note_id} ("
                f"{', '.join(repr(d.text) for d in discarded)}) because they "
                f"were overlapping with other spans with the same label."
            )

        return {
            **self.preprocess(doc),
            "targets": tags,
        }

    def collate(self, preps) -> NERBatchInput:
        collated: NERBatchInput = {
            "embedding": self.embedding.collate(preps["embedding"]),
        }
        max_len = max(preps["length"])
        if "targets" in preps:
            targets = torch.as_tensor(
                [
                    row + [[-1] * len(self.labels)] * (max_len - len(row))
                    for sample_targets in preps["targets"]
                    for row in sample_targets
                ],
                dtype=torch.long,
            )
            # targets = (targets.unsqueeze(-1) == torch.arange(5)).to(device)
            # mask = (targets[:, 0] != -1).to(device)
            collated["targets"] = targets
        else:
            if self.window > 1:
                win_indices, win_indexer = make_windows(
                    preps["length"],
                    self.window,
                    self.stride,
                )
                collated["window_indices"] = win_indices
                collated["window_indexer"] = win_indexer
        return collated

    def forward(self, batch: NERBatchInput) -> NERBatchOutput:
        encoded = self.embedding.module_forward(batch["embedding"])
        embeddings = encoded["embeddings"]
        mask = encoded["mask"]
        # batch words (labels tags) -> batch words labels tags
        num_samples, num_words = embeddings.shape[:-1]
        num_labels = len(self.labels)
        scores = self.linear(embeddings).view((num_samples, num_words, num_labels, 5))
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
            if self.window == 1:
                tags = scores.argmax(-1).masked_fill(~mask.unsqueeze(-1), 0)
            elif self.window <= 0:
                tags = self.crf.decode(scores, mask)
            else:
                win_scores = scores.view(num_samples * num_words, num_labels, 5)[
                    batch["window_indices"]
                ]
                win_tags = self.crf.decode(win_scores, batch["window_indices"] != -1)
                tags = win_tags.view(win_tags.shape[0] * win_tags.shape[1], num_labels)[
                    batch["window_indexer"]
                ]
                tags = tags.view(num_samples, num_words, num_labels)
                tags = tags.masked_fill(~mask.unsqueeze(-1), 0)

            # tags = scores.argmax(-1).masked_fill(~mask.unsqueeze(-1), 0)
        if loss is not None and loss.item() > 100000:
            warnings.warn("The loss is very high, this is likely a tag encoding issue.")
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
        spans: Dict[Doc, list[Span]] = defaultdict(list)
        embedded_spans = [
            span
            for doc in docs
            for span in list(get_spans(doc, self.embedding.span_getter))
        ]
        for embedded_span_idx, label_idx, start, end in self.crf.tags_to_spans(
            batch["tags"].cpu()
        ).tolist():
            span = embedded_spans[embedded_span_idx][start:end]
            span.label_ = self.labels[label_idx]
            spans[span.doc].append(span)
        for doc in docs:
            self.set_spans(doc, spans.get(doc, []))
        return docs
