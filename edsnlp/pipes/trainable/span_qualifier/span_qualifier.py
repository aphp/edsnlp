from __future__ import annotations

import os
import pickle
import warnings
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
)

import foldedtensor as ft
import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import NotRequired, TypedDict

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
)
from edsnlp.utils.bindings import (
    BINDING_GETTERS,
    BINDING_SETTERS,
    Binding,
    Qualifiers,
    QualifiersArg,
)
from edsnlp.utils.span_getters import get_spans

SpanQualifierBatchInput = TypedDict(
    "SpanQualifierBatchInput",
    {
        "embedding": BatchInput,
        "bindings_group_mask": torch.Tensor,
        "bindings": NotRequired[torch.Tensor],
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
mask: torch.BoolTensor
    Mask of the sequences
spans: torch.LongTensor
    2d tensor of n_spans * (doc_idx, ner_label_idx, begin, end)
targets: NotRequired[List[torch.LongTensor]]
    list of 2d tensor of n_spans * n_combinations (1 hot)
"""


class TrainableSpanQualifier(
    TorchComponent[BatchOutput, SpanQualifierBatchInput], BaseComponent
):
    """
    The `eds.span_qualifier` component is a trainable qualifier, predictor of span
    attributes. In this context, the span qualification task consists in assigning
    values (boolean, strings or any complex object) to attributes/extensions of spans
    such as:

    - `span.label_`,
    - `span._.negation`,
    - `span._.date.mode`
    - etc.

    Architecture
    ------------
    The underlying `eds.span_multilabel_classifier.v1` model performs span
    classification by:

    1. Pooling the words embedding (`mean`, `max` or `sum`) into a single embedding per
    span
    2. Computing logits for each possible binding (i.e. qualifier-value assignment)
    3. Splitting these bindings into independent groups such as

        - `event=start` and `event=stop`
        - `negated=False` and `negated=True`

    4. Learning or predicting a combination amongst legal combination of these bindings.
    For instance in the second group, we can't have both `negated=True` and
    `negated=False` so the combinations are `[(1, 0), (0, 1)]`
    5. Assigning bindings on spans depending on the predicted results

    Step by step
    ------------
    ## Initialization
    During the initialization of the pipeline, the `span_qualifier` component will
    gather all spans that match `on_ents` and `on_span_groups` patterns (or
    `candidate_getter` function). It will then list all possible values for each
    `qualifier` of the `qualifiers` list and store every possible (qualifier, value)
    pair (i.e. binding).

    For instance, a custom qualifier `negation` with possible values `True` and `False`
    will result in the following bindings
    `[("_.negation", True), ("_.negation", False)]`, while a custom qualifier
    `event` with possible values `start`, `stop`, and `start-stop` will result in
    the following bindings
    `[("_.event", "start"), ("_.event", "stop"), ("_.event", "start-stop")]`.

    ## Training
    During training, the `span_qualifier` component will gather spans on the documents
    in a mini-batch and evaluate each binding on each span to build a supervision
    matrix. This matrix will be feed it to the underlying model (most likely a
    `eds.span_multilabel_classifier.v1`). The model will compute logits for each entry
    of the matrix and compute a cross-entropy loss for each group of bindings sharing
    the same qualifier.

    ## Prediction
    During prediction, the `span_qualifier` component will gather spans on a given
    document and evaluate each binding on each span using the underlying model. Using
    the same binding exclusion and label constraint mechanisms as during training,
    scores will be computed for each binding and the best legal combination of bindings
    will be selected. Finally, the selected bindings will be assigned to the spans.

    Examples
    --------
    Let us define the pipeline and train it. We provide utils to train the model using
    an API, but you can use a spaCy's config file as well.

    ```python
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
        "eds.span_qualifier",
        name="qualifier",
        config={
            "embedding": {
                "@factory": "eds.span_pooler",
                "embedding": nlp.get_pipe("transformer"),
                "span_getter": ["ents", "sc"],
            },
            "qualifiers": ["_.negation", "_.event_type"],
        },
    )
    ```

    Parameters
    ----------
    nlp: PipelineProtocol
        Spacy vocabulary
    name: str
        Name of the component
    embedding : SpanEmbeddingComponent
        The word embedding component
    qualifiers: QualifiersArg
        The qualifiers to predict or train on. If a dict is given, keys are the
        qualifiers and values are the labels for which the qualifier is allowed, or True
        if the qualifier is allowed for all labels.
    keep_none: bool
        If False, skip spans for which a qualifier returns None. If True (default), the
        None values will be learned and predicted, just as any other value.
    """

    qualifiers: Qualifiers

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "eds.span_qualifier",
        *,
        embedding: SpanEmbeddingComponent,
        qualifiers: QualifiersArg,
        keep_none: bool = False,
    ):
        self.qualifiers = {
            (k if k.startswith("_.") else f"_.{k}"): v for k, v in qualifiers.items()
        }

        super().__init__(nlp, name)

        self.embedding = embedding

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.classifier = torch.nn.Linear(embedding.output_size, 0)

        self.bindings_indexer_per_group: List[slice] = []
        self.bindings_group_mask = None
        self.bindings: List[Binding] = []
        self.group_qualifiers: List[set] = []
        self.keep_none = keep_none

    @property
    def span_getter(self):
        return self.embedding.span_getter

    def to_disk(self, path, *, exclude=set()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        # We save the bindings as a pickle file since values can be arbitrary objects
        os.makedirs(path, exist_ok=True)
        data_path = path / "bindings.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "bindings": self.bindings,
                    "bindings_indexer_per_group": self.bindings_indexer_per_group,
                    "combinations": [
                        comb.tolist() for comb in self.combinations_per_group
                    ],
                },
                f,
            )
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path, exclude=tuple()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        data_path = path / "bindings.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.bindings = data["bindings"]
        self.bindings_indexer_per_group = data["bindings_indexer_per_group"]
        self.group_qualifiers = [
            list(dict.fromkeys(key for key, _ in self.bindings[group_indexer]))
            for group_indexer in self.bindings_indexer_per_group
        ]
        self.classifier = torch.nn.Linear(
            self.embedding.output_size,
            len(self.bindings),
        )
        for grp_idx, combinations in enumerate(data["combinations"]):
            self.register_buffer(
                f"combinations_{grp_idx}",
                torch.tensor(combinations, dtype=torch.float),
            )
        self.set_extensions()
        super().from_disk(path, exclude=exclude)

    def set_extensions(self):
        super().set_extensions()
        for qlf in self.qualifiers or ():
            if qlf.startswith("_."):
                qlf = qlf[2:]
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)

        qualifier_values = defaultdict(set)
        for doc in gold_data:
            spans = list(get_spans(doc, self.embedding.span_getter))
            for span in spans:
                for qualifier, labels in self.qualifiers.items():
                    if labels is True or span.label_ in labels:
                        value = BINDING_GETTERS[qualifier](span)
                        if value is not None or self.keep_none:
                            qualifier_values[qualifier].add(value)

        qualifier_values = {
            key: sorted(values, key=str) for key, values in qualifier_values.items()
        }

        for qualifier, values in qualifier_values.items():
            if len(values) < 2:
                warnings.warn(
                    f"Qualifier {qualifier} should have at least 2 values, found "
                    f"{len(values)}: {values}"
                )

        # if self.qualifiers is not None and set(self.qualifiers) != set(
        #     qualifier_values.keys()
        # ):
        #     warnings.warn(
        #         f"Qualifiers {sorted(self.qualifiers)} do not match qualifiers found
        #         f"in gold data {sorted(qualifier_values.keys())}"
        #     )
        self.bindings = []
        self.bindings_indexer_per_group = []

        # `groups_combinations`:
        # - for each group (i.e. qualifier here)
        # - for each legal combination of bindings in this group
        # - bindings in this combination
        # ATM, create one group per qualifier (e.g. one group for _.event,
        # one for _.negation, ...), and so one binding per combination
        combinations_per_group = [
            [((key, value),) for value in sorted(values, key=str)]
            for key, values in qualifier_values.items()
        ]
        bindings_per_group = [
            list(dict.fromkeys(binding for comb in group for binding in comb))
            for group in combinations_per_group
        ]
        self.group_qualifiers = [
            list(dict.fromkeys(key for key, _ in group)) for group in bindings_per_group
        ]

        for grp_idx, (group_combinations, group_bindings) in enumerate(
            zip(combinations_per_group, bindings_per_group)
        ):
            self.bindings_indexer_per_group.append(
                slice(len(self.bindings), len(self.bindings) + len(group_bindings))
            )
            self.bindings.extend(group_bindings)
            # combinations buffer: bool tensor of shape
            #   num legal combinations
            # * num bindings in this group
            # TODO: use sparse tensor or None to mark identity combinations
            self.register_buffer(
                f"combinations_{grp_idx}",
                ft.as_folded_tensor(
                    [
                        [binding in combination_bindings for binding in group_bindings]
                        for combination_bindings in group_combinations
                    ],
                    dtype=torch.float,
                    full_names=("combination", "binding"),
                ).as_tensor(),
            )
        if len(self.bindings) == 0:
            raise ValueError(
                f"No bindings found for qualifiers {sorted(self.qualifiers)} on the "
                f"spans provided by the span embedding ({self.embedding.span_getter})."
            )
        self.classifier = torch.nn.Linear(
            in_features=self.classifier.in_features,
            out_features=len(self.bindings),
        )

    @property
    def combinations_per_group(self):
        return (
            getattr(self, f"combinations_{i}")
            for i in range(len(self.bindings_indexer_per_group))
        )

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        embedding = self.embedding.preprocess(doc)
        # "$" prefix means this field won't be accessible from the outside.
        spans = embedding["$spans"]
        return {
            "embedding": embedding,
            "bindings_group_mask": [
                [
                    not set(group_qualifiers).isdisjoint(
                        set(
                            qlf
                            for qlf, span_filter in self.qualifiers.items()
                            if span_filter is True or span.label_ in span_filter
                        )
                    )
                    for group_qualifiers in self.group_qualifiers
                ]
                for span in spans
            ],
        }

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        preps = self.preprocess(doc)
        spans = preps["embedding"]["$spans"]
        return {
            **preps,
            "bindings": [
                [
                    (
                        self.qualifiers[binding[0]] is True
                        or span.label_ in self.qualifiers[binding[0]]
                    )
                    and BINDING_GETTERS[binding](span)
                    for binding in self.bindings
                ]
                for span in spans
            ],
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanQualifierBatchInput:
        bindings_group_mask = ft.as_folded_tensor(
            batch["bindings_group_mask"],
            dtype=torch.bool,
            full_names=("sample", "span", "binding"),
            data_dims=("span", "binding"),
        ).as_tensor()
        bindings_group_mask = bindings_group_mask.view(
            len(bindings_group_mask), len(self.group_qualifiers)
        )
        collated: SpanQualifierBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "bindings_group_mask": bindings_group_mask,
        }
        if "bindings" in batch:
            bindings = ft.as_folded_tensor(
                [
                    span_bindings
                    for sample in batch["bindings"]
                    for span_bindings in sample
                ],
                dtype=torch.float,
                full_names=("span", "binding"),
            ).as_tensor()
            collated["bindings"] = bindings.view(len(bindings), len(self.bindings))
        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanQualifierBatchInput) -> BatchOutput:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans
        If labels are predicted, they are assigned to the `additional_outputs`
        dictionary.

        Parameters
        ----------
        batch: SpanQualifierBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        embedding = self.embedding.module_forward(batch["embedding"])
        span_embeds = embedding["embeddings"]

        n_spans, dim = span_embeds.shape
        n_bindings = len(self.bindings)
        device = span_embeds.device

        binding_scores = self.classifier(span_embeds)

        if "bindings" not in batch:
            pred = torch.zeros(n_spans, n_bindings, device=device)
            losses = None
        else:
            losses = [torch.zeros((), dtype=torch.float, device=device)]
            pred = None

        for group_idx, (bindings_indexer, combinations) in enumerate(
            zip(
                self.bindings_indexer_per_group,
                self.combinations_per_group,
            )
        ):
            group_samples_mask = batch["bindings_group_mask"][:, group_idx]
            combinations_scores = torch.einsum(
                "eb,cb->ec",
                binding_scores[group_samples_mask][:, bindings_indexer],
                combinations.float(),
            )

            if "bindings" in batch:
                group_samples_mask = batch["bindings_group_mask"][:, group_idx]
                # ([e]ntities * [b]indings) * ([c]ombinations * [b]indings)
                targets = torch.einsum(
                    "eb,cb->ec",
                    batch["bindings"][group_samples_mask][:, bindings_indexer],
                    combinations,
                )
                # [e]ntities * [c]comb --(argmax)--> [e]entities
                # Choose the combination of bindings in this group that fits the
                # most the gold bindings
                targets = targets.argmax(-1)
                losses.append(
                    F.cross_entropy(combinations_scores, targets, reduction="sum")
                )
                assert not torch.isnan(losses[-1]).any(), combinations_scores
            elif "bindings" not in batch:
                pred[:, bindings_indexer][group_samples_mask] = combinations[
                    combinations_scores.argmax(-1)
                ]

        if "bindings" in batch:
            assert len(losses) > 0, "No group found"

        return {
            "loss": sum(losses) if losses is not None else None,
            "labels": pred,
        }

    def postprocess(self, docs: Sequence[Doc], batch: BatchOutput) -> Sequence[Doc]:
        spans = [
            span for doc in docs for span in get_spans(doc, self.embedding.span_getter)
        ]
        for span_idx, binding_idx in batch["labels"].nonzero(as_tuple=False):
            span = spans[span_idx]
            BINDING_SETTERS[self.bindings[binding_idx]](span)
        return docs
