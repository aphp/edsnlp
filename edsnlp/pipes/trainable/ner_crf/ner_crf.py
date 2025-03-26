from __future__ import annotations

import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp.core.pipeline import Pipeline
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
    get_spans_with_group,
)
from edsnlp.utils.torch import make_windows

NERBatchInput = TypedDict(
    "NERBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
        "window_indices": NotRequired[torch.Tensor],
        "window_indexer": NotRequired[torch.Tensor],
        "stats": Dict[str, int],
    },
)
NERBatchOutput = TypedDict(
    "NERBatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "tags": Optional[torch.Tensor],
        "probs": Optional[torch.Tensor],
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
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.ner_crf(
            embedding=eds.transformer(
                model="prajjwal1/bert-tiny",
                window=128,
                stride=96,
            ),
            mode="joint",
            target_span_getter="ner-gold",
            span_setter="ents",
            window=10,
        ),
        name="ner"
    )
    ```

    To train the model, refer to the [Training](/tutorials/training)
    tutorial.

    Extensions
    ----------

    !!! warning "Experimental Confidence Score"

        The NER confidence score feature is experimental and the API and underlying
        algorithm may change.

    The `eds.ner_crf` pipeline declares one extension on the `Span` object:

    - `span._.ner_confidence_score`: The confidence score of the Named Entity
    Recognition (NER) model for the given span.

    The `ner_confidence_score` is computed based on the Average Entity Confidence
    Score using the following formula:

    $$
    \\text{Average Entity Confidence Score} =
    \\frac{1}{n} \\sum_{i \\in \\text{tokens}} (1 - p(O)_i)
    $$

    Where:

    - $n$ is the number of tokens.
    - $\\text{tokens}$ refers to the tokens within the span.
    - $p(O)_i$ represents the probability of token $i$ belonging to class 'O'
    (Outside entity).

    !!! warning "Confidence score is not computed by default"
        By default, the confidence score is not computed, as it adds around 5% to
        inference time. You can enable its computation with:
        ```python
        nlp.pipes.ner.compute_confidence_score = True
        ```

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
    context_getter : Optional[SpanGetterArg]
        What context to use when computing the span embeddings (defaults to the whole
        document). For example `{"section": "conclusion"}` to only extract the
        entities from the conclusion.
    mode : Literal["independent", "joint", "marginal"]
        The CRF mode to use : independent, joint or marginal
    window : int
        The window size to use for the CRF. If 0, will use the whole document, at
        the cost of a longer computation time. If 1, this is equivalent to assuming
        that the tags are independent and will the component be faster, but with
        degraded performance. Empirically, we found that a window size of 10 or 20
        works well.
    stride : Optional[int]
        The stride to use for the CRF windows. Defaults to `window // 2`.

    Authors and citation
    --------------------
    The `eds.ner_crf` pipeline was developed by AP-HP's Data Science team.

    The deep learning model was adapted from [@wajsburt:tel-03624928].
    """

    span_setter: SpanSetter

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: Optional[str] = "ner_crf",
        *,
        embedding: WordEmbeddingComponent,
        target_span_getter: SpanGetterArg = {"ents": True},
        context_getter: Optional[SpanGetterArg] = None,
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
        sub_context_getter = getattr(embedding, "context_getter", None)
        if (
            sub_context_getter is not None and context_getter is None
        ):  # pragma: no cover
            context_getter = sub_context_getter

        super().__init__(
            nlp=nlp,
            name=name,
            span_setter=span_setter or {},
        )
        self.context_getter = context_getter
        self.infer_span_setter = (
            span_setter is None if infer_span_setter is None else infer_span_setter
        )

        self.embedding = embedding
        self.labels = labels
        self.labels_to_idx = {lab: i for i, lab in enumerate(labels)} if labels else {}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Initializing zero-element tensors is a no-op"
            )
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

        self.compute_confidence_score: bool = False

    def set_extensions(self) -> None:
        """
        Set spaCy extensions
        """
        super().set_extensions()
        if not Span.has_extension("ner_confidence_score"):
            Span.set_extension("ner_confidence_score", default={})

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
                for span in get_spans(doc, self.target_span_getter):
                    inferred_labels.add(span.label_)
            else:
                for span, key in get_spans_with_group(doc, self.target_span_getter):
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
        self.linear.out_features = new_linear.out_features
        self.linear.bias = new_linear.bias

        # Update initialization arguments
        self.labels = labels
        self.labels_to_idx = {lab: i for i, lab in enumerate(labels)}

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

    def preprocess(self, doc, **kwargs):
        if self.labels is None:
            raise ValueError(
                "The component was not initialized with any labels. Please "
                "initialize it with the `labels` parameter, pass a list of "
                "labels to the `update_labels` method, or call the `post_init` "
                "method with a list of documents."
            )

        ctxs = get_spans(doc, self.context_getter) if self.context_getter else [doc[:]]
        lengths = [len(ctx) for ctx in ctxs]
        return {
            "embedding": self.embedding.preprocess(doc, contexts=ctxs, **kwargs),
            "lengths": lengths,
            "$contexts": ctxs,
            "stats": {"ner_words": sum(lengths)},
        }

    def preprocess_supervised(self, doc, **kwargs):
        prep = self.preprocess(doc, **kwargs)
        contexts = prep["$contexts"]
        tags = []

        discarded = []
        for context, target_ents in zip(
            contexts,
            align_spans(
                list(get_spans(doc, self.target_span_getter)),
                contexts,
            ),
        ):
            span_tags = [[0] * len(self.labels) for _ in range(len(context))]
            start = context.start
            by_label = defaultdict(list)
            for ent in target_ents:
                if ent.label_ in self.labels_to_idx:
                    label_idx = self.labels_to_idx.get(ent.label_)
                    by_label[label_idx].append(ent)
            filtered = []
            for label_idx, spans in by_label.items():
                filtered[len(filtered) :], discarded[len(discarded) :] = filter_spans(
                    by_label[label_idx], return_discarded=True
                )

            for ent in filtered:
                label_idx = self.labels_to_idx[ent.label_]
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
                "Some spans were discarded in the training data because they "
                "were overlapping with other spans with the same label."
            )

        return {
            **prep,
            "targets": tags,
        }

    def collate(self, preps) -> NERBatchInput:
        collated: NERBatchInput = {
            "embedding": self.embedding.collate(preps["embedding"]),
            "stats": {
                k: sum(v) for k, v in preps["stats"].items() if not k.startswith("__")
            },
        }
        lengths = [length for sample in preps["lengths"] for length in sample]
        max_len = max(lengths)
        if "targets" in preps:
            targets = torch.as_tensor(
                [
                    row + [[-1] * len(self.labels)] * (max_len - len(row))
                    for sample_targets in preps["targets"]
                    for row in sample_targets
                ],
                dtype=torch.long,
            )
            collated["targets"] = targets
        else:
            if self.window > 1:
                win_indices, win_indexer = make_windows(
                    lengths,
                    self.window,
                    self.stride,
                )
                collated["window_indices"] = win_indices
                collated["window_indexer"] = win_indexer
        return collated

    def forward(self, batch: NERBatchInput) -> NERBatchOutput:
        embeddings = self.embedding(batch["embedding"])["embeddings"]
        embeddings = embeddings.refold("context", "word")
        mask = embeddings.mask
        # batch words (labels tags) -> batch words labels tags
        num_contexts, num_words = embeddings.shape[:-1]
        num_labels = len(self.labels)
        scores = self.linear(embeddings).view((num_contexts, num_words, num_labels, 5))
        probs = None
        if self.compute_confidence_score:
            probs = torch.nn.functional.softmax(scores, dim=-1)
        loss = tags = None
        if "targets" in batch:
            if self.mode == "independent":
                loss = (
                    torch.nn.functional.cross_entropy(
                        scores.view(-1, 5),
                        batch["targets"].view(-1),
                        ignore_index=-1,
                        reduction="sum",
                    )
                    / batch["stats"]["ner_words"]
                )
            elif self.mode == "joint":
                loss = (
                    self.crf(
                        scores,
                        mask,
                        batch["targets"].unsqueeze(-1)
                        == torch.arange(5).to(scores.device),
                    ).sum()
                    / batch["stats"]["ner_words"]
                )
            elif self.mode == "marginal":
                loss = (
                    torch.nn.functional.cross_entropy(
                        self.crf.marginal(
                            scores,
                            mask,
                        ).view(-1, 5),
                        batch["targets"].view(-1),
                        ignore_index=-1,
                        reduction="sum",
                    )
                    / batch["stats"]["ner_words"]
                )
        else:
            if self.window == 1:
                tags = scores.argmax(-1).masked_fill(~mask.unsqueeze(-1), 0)
            elif self.window <= 0:
                tags = self.crf.decode(scores, mask)
            else:
                win_scores = scores.view(num_contexts * num_words, num_labels, 5)[
                    batch["window_indices"]
                ]
                win_tags = self.crf.decode(win_scores, batch["window_indices"] != -1)
                tags = win_tags.view(win_tags.shape[0] * win_tags.shape[1], num_labels)[
                    batch["window_indexer"]
                ]
                tags = tags.view(num_contexts, num_words, num_labels)
                tags = tags.masked_fill(~mask.unsqueeze(-1), 0)

            # tags = scores.argmax(-1).masked_fill(~mask.unsqueeze(-1), 0)
        if loss is not None and loss.item() > 100000:
            warnings.warn("The loss is very high, this is likely a tag encoding issue.")
        return {
            "loss": loss,
            "tags": tags,
            "probs": probs,
        }

    def postprocess(
        self,
        docs: List[Doc],
        results: NERBatchOutput,
        inputs: List[Dict[str, Any]],
    ):
        spans: Dict[Doc, list[Span]] = defaultdict(list)
        contexts = [ctx for sample in inputs for ctx in sample["$contexts"]]
        tags = results["tags"]
        if self.compute_confidence_score:
            probs = results["probs"]

        for ctx, label, start, end in self.crf.tags_to_spans(tags).tolist():
            span = contexts[ctx][start:end]
            span.label_ = self.labels[label]
            if self.compute_confidence_score:
                span_probs = probs[ctx, start:end, label, :]
                average_entity_confidence_score = torch.mean(
                    1 - span_probs[:, 0]
                ).item()
                span._.ner_confidence_score = average_entity_confidence_score

            spans[span.doc].append(span)
        for doc in docs:
            self.set_spans(doc, spans.get(doc, []))
        return docs
