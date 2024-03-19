from __future__ import annotations

import logging
import os
import pickle
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
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
    Qualifiers,
    QualifiersArg,
)
from edsnlp.utils.span_getters import SpanFilter, get_spans

logger = logging.getLogger(__name__)

SpanQualifierBatchInput = TypedDict(
    "SpanQualifierBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
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
    The `eds.span_qualifier` component is a trainable qualifier predictor. In EDS-NLP,
    we call span attributes "qualifiers". In this context, the span qualification task
    consists in assigning values (boolean, strings or any complex object) to
    attributes/extensions of spans such as:

    - `span._.negation`,
    - `span._.date.mode`
    - `span._.cui`

    In the rest of this page, we will refer to a pair of (qualifier, value) as a
    "binding". For instance, the binding `("_.negation", True)` means that the
    qualifier `negation` of the span is (or should be, when predicted) set to `True`.

    Architecture
    ------------
    The model performs span classification by:

    1. Calling a word pooling embedding such as `eds.span_pooler` to compute a single
    embedding for each span
    2. Computing logits for each possible binding using a linear layer
    3. Splitting these bindings into groups of exclusive values such as

        - `event=start` and `event=stop`
        - `negated=False` and `negated=True`

        Note that the above groups are not exclusive, but the values within each group
        are.

    4. Applying the best scoring binding in each group to each span

    Examples
    --------
    To create a span qualifier component, you can use the following code:

    ```python
    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.span_qualifier",
        name="qualifier",
        config={
            "embedding": {
                # To embed the spans, we will use a span pooler
                "@factory": "eds.span_pooler",
                "pooling_mode": "mean",  # mean pooling
                "span_getter": ["ents", "sc"],
                "embedding": {
                    # that will use a transformer to embed the doc words
                    "@factory": "eds.transformer",
                    "model": "prajjwal1/bert-tiny",
                    "window": 128,
                    "stride": 96,
                },
            },
            # For every span embedded by the span pooler
            # (doc.ents and doc.spans["sc"]), we will predict both
            # span._.negation and span._.event_type
            "qualifiers": ["_.negation", "_.event_type"],
        },
    )
    ```

    To infer the values of the qualifiers, you can use the pipeline `post_init` method:

    ```{ .python .no-check }
    nlp.post_init(gold_data)
    ```

    To train the model, refer to the [Training](/tutorials/training.md) tutorial.

    You can inspect the bindings that will be used for training and prediction
    ```{ .python .no-check }
    print(nlp.pipes.qualifier.bindings)
    # list of (qualifier name, span labels or True if all, values)
    # Out: [
    #   ('_.negation', True, [True, False]),
    #   ('_.event_type', True, ['start', 'stop'])
    # ]
    ```

    You can also change these values and update the bindings by calling the
    `update_bindings` method. Don't forget to retrain the model if new values are
    added !

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    embedding : SpanEmbeddingComponent
        The word embedding component
    qualifiers : QualifiersArg
        The qualifiers to predict or train on. If a dict is given, keys are the
        qualifiers and values are the labels for which the qualifier is allowed, or True
        if the qualifier is allowed for all labels.
    keep_none : bool
        If False, skip spans for which a qualifier returns None. If True (default), the
        None values will be learned and predicted, just as any other value.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "eds.span_qualifier",
        *,
        embedding: SpanEmbeddingComponent,
        qualifiers: QualifiersArg,
        values: Optional[Dict[str, List[Any]]] = None,
        keep_none: bool = False,
    ):
        qualifiers: Qualifiers
        self.values = values
        self.keep_none = keep_none
        self.bindings: List[Tuple[str, List[str], List[Any]]] = [
            (k if k.startswith("_.") else f"_.{k}", v, [])
            for k, v in qualifiers.items()
        ]

        super().__init__(nlp, name)
        self.embedding = embedding
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.classifier = torch.nn.Linear(embedding.output_size, 0)

    @property
    def qualifiers(self) -> Qualifiers:
        return {qlf: labels for qlf, labels, _ in self.bindings}

    @qualifiers.setter
    def qualifiers(self, value: Qualifiers):
        bindings = []
        for qlf, labels in value.items():
            groups = [group for group in self.bindings if group[0] == qlf]
            if len(groups) > 1:
                raise ValueError(
                    f"Qualifier {qlf} has different label filters: "
                    f"{[g[0] for g in groups]}. Please use the `update_bindings` "
                    f"method to update the labels."
                )
            if groups:
                bindings.append((qlf, labels, groups[0][2]))
        self.update_bindings(bindings)

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
        self.update_bindings(data["bindings"])
        self.set_extensions()
        super().from_disk(path, exclude=exclude)

    @property
    def bindings_to_idx(self) -> List[Tuple[str, List[str], Dict[Any, int]]]:
        if getattr(self, "_bindings_to_idx", None) is not None:
            return self._bindings_to_idx
        self._bindings_to_idx = [  # noqa
            (qualifier, labels, {value: idx for idx, value in enumerate(values)})
            for qualifier, labels, values in self.bindings
        ]
        return self._bindings_to_idx

    @property
    def bindings_indexers(self) -> List[Union[Sequence[int], slice]]:
        return self._bindings_indexers

    def set_extensions(self):
        super().set_extensions()
        for group in self.bindings:
            qlf = group[0]
            if qlf.startswith("_."):
                qlf = qlf[2:]
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)

        bindings = [
            (qlf, labels, dict.fromkeys(vals)) for qlf, labels, vals in self.bindings
        ]
        for doc in gold_data:
            spans = list(get_spans(doc, self.embedding.span_getter))
            for span in spans:
                for qualifier, labels, values in bindings:
                    if labels is True or span.label_ in labels:
                        value = BINDING_GETTERS[qualifier](span)
                        if value is not None or self.keep_none:
                            values[value] = None

        bindings = [
            (qualifier, labels, sorted(values, key=str))
            for qualifier, labels, values in bindings
        ]

        for qualifier, labels, values in bindings:
            if len(values) < 2:
                warnings.warn(
                    f"Qualifier {qualifier} for labels {labels} should have at "
                    f"least 2 values but found {len(values)}: {values}."
                )

        self.update_bindings(bindings)

    def update_bindings(self, bindings: List[Tuple[str, SpanFilter, List[Any]]]):
        keep_bindings = [
            (labels is True or len(labels) > 0) and len(values) > 0
            for k, labels, values in bindings
        ]
        if not all(keep_bindings):
            logger.warning(
                "Some qualifiers have no labels or values and have been removed:"
                + "".join(
                    "\n- " + str(b[0]) + " for labels " + str(b[1])
                    for b, keep in zip(bindings, keep_bindings)
                    if not keep
                )
            )
        bindings = [b for b, keep in zip(bindings, keep_bindings) if keep]
        new_bindings = list(
            dict.fromkeys((k, v) for k, _, group in bindings for v in group)
        )
        old_bindings = list(
            dict.fromkeys((k, v) for k, _, group in self.bindings for v in group)
        )

        new_bindings_to_idx = {binding: idx for idx, binding in enumerate(new_bindings)}
        old_bindings_to_idx = {binding: idx for idx, binding in enumerate(old_bindings)}

        common = [b for b in new_bindings if b in old_bindings]
        old_index = [old_bindings_to_idx[label] for label in common]
        new_index = [new_bindings_to_idx[label] for label in common]
        new_linear = torch.nn.Linear(self.classifier.in_features, len(new_bindings))
        new_linear.weight.data[new_index] = self.classifier.weight.data[old_index]
        self.classifier.weight = new_linear.weight
        self.classifier.out_features = new_bindings
        missing_bindings = set(new_bindings) - set(old_bindings)
        if missing_bindings and len(old_bindings) > 0:
            warnings.warn(
                f"Added {len(missing_bindings)} new bindings. Consider retraining "
                f"the model to learn these new bindings."
            )

        if hasattr(self.classifier, "bias"):
            new_linear.bias.data[new_index] = self.classifier.bias.data[old_index]
            self.classifier.bias = new_linear.bias

        def simplify_indexer(indexer):
            return (
                slice(indexer[0], indexer[-1] + 1)
                if len(indexer) and list(range(indexer[0], indexer[-1] + 1)) == indexer
                else indexer
            )

        self.bindings = bindings
        self._bindings_indexers = [
            simplify_indexer([new_bindings_to_idx[(qlf, value)] for value in values])
            for qlf, labels, values in bindings
        ]
        self._bindings_to_idx = None

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        return {
            "embedding": self.embedding.preprocess(doc),
        }

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        preps = self.preprocess(doc)
        spans = preps["embedding"]["$spans"]
        return {
            **preps,
            "targets": [
                [
                    values_to_idx.get(BINDING_GETTERS[qlf](span), -100)
                    if labels is True or span.label_ in labels
                    else -100
                    for qlf, labels, values_to_idx in self.bindings_to_idx
                ]
                for span in spans
            ],
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanQualifierBatchInput:
        collated: SpanQualifierBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
        }
        if "targets" in batch:
            targets = ft.as_folded_tensor(
                batch["targets"],
                dtype=torch.long,
                full_names=("sample", "span", "group"),
                data_dims=("span", "group"),
            ).as_tensor()
            collated["targets"] = targets.view(len(targets), len(self.bindings))
        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanQualifierBatchInput) -> BatchOutput:
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
        embedding = self.embedding.module_forward(batch["embedding"])
        span_embeds = embedding["embeddings"]

        binding_scores = self.classifier(span_embeds)

        if "targets" in batch:
            losses = []
            pred = None
        else:
            pred = []
            losses = None

        # For each group, for instance:
        # - `event=start` and `event=stop`
        # - `negated=False` and `negated=True`
        for group_idx, bindings_indexer in enumerate(self.bindings_indexers):
            if "targets" in batch:
                losses.append(
                    F.cross_entropy(
                        binding_scores[:, bindings_indexer],
                        batch["targets"][:, group_idx],
                        reduction="sum",
                    )
                )
                assert not torch.isnan(losses[-1]).any(), "NaN loss"
            else:
                pred.append(binding_scores[:, bindings_indexer].argmax(dim=1))

        return {
            "loss": sum(losses) if losses is not None else None,
            "labels": pred,
        }

    def postprocess(self, docs: Sequence[Doc], batch: BatchOutput) -> Sequence[Doc]:
        # Preprocessed docs should still be in the cache
        spans = [s for doc in docs for s in self.preprocess(doc)["embedding"]["$spans"]]
        # For each prediction group (exclusive bindings)...
        for val_indices, (qlf, labels, values) in zip(batch["labels"], self.bindings):
            # For each span...
            for span, idx in zip(spans, val_indices.tolist()):
                # If the span is not filtered out...
                if labels is True or span.label_ in labels:
                    # ...assign the predicted value to the span
                    BINDING_SETTERS[qlf](span, values[idx])
        return docs
