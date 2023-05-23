import pickle
from collections import defaultdict
from itertools import islice
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import spacy
from spacy import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from thinc.api import Model, Optimizer
from thinc.backends import NumpyOps
from thinc.model import set_dropout_rate
from thinc.types import Ints2d
from wasabi import Printer

from edsnlp.pipelines.trainable.span_qualifier.utils import (
    Binding,
    SpanGroups,
    Spans,
    keydefaultdict,
    make_binding_getter,
    make_binding_setter,
)

NUM_INITIALIZATION_EXAMPLES = 10

msg = Printer()
np_ops = NumpyOps()


@spacy.registry.scorers("eds.span_qualifier_scorer.v1")
def make_span_qualifier_scorer(candidate_getter: Callable):
    def span_qualifier_scorer(examples: Iterable[Example], **cfg):
        """
        Scores the extracted entities that may be overlapping or nested
        by looking in `doc.ents`, and `doc.spans`.

        Parameters
        ----------
        examples: Iterable[Example]
            The examples to score
        cfg: Dict[str]
            The configuration dict of the component

        Returns
        -------
        Dict[str, float]
        """
        labels = defaultdict(lambda: ([], []))
        labels["ALL"] = ([], [])
        for eg_idx, eg in enumerate(examples):
            doc_spans, *_, doc_qlf = candidate_getter(eg.predicted)
            for span_idx, (span, span_qualifiers) in enumerate(zip(doc_spans, doc_qlf)):
                for qualifier in span_qualifiers:
                    value = BINDING_GETTERS[qualifier](span)
                    if value:
                        labels["ALL"][0].append((eg_idx, span_idx, qualifier, value))
                        key_str = f"{qualifier}" if value is True else f"{value}"
                        labels[key_str][0].append((eg_idx, span_idx, value))

            doc_spans, *_, doc_qlf = candidate_getter(eg.reference)
            for span_idx, (span, span_qualifiers) in enumerate(zip(doc_spans, doc_qlf)):
                for qualifier in span_qualifiers:
                    value = BINDING_GETTERS[qualifier](span)
                    if value:
                        labels["ALL"][1].append((eg_idx, span_idx, qualifier, value))
                        key_str = f"{qualifier}" if value is True else f"{value}"
                        labels[key_str][1].append((eg_idx, span_idx, value))

        def prf(pred, gold):
            tp = len(set(pred) & set(gold))
            np = len(pred)
            ng = len(gold)
            return {
                "f": 2 * tp / max(1, np + ng),
                "p": 1 if tp == np else (tp / np),
                "r": 1 if tp == ng else (tp / ng),
            }

        results = {name: prf(pred, gold) for name, (pred, gold) in labels.items()}
        return {"qual_f": results["ALL"]["f"], "qual_per_type": results}

    return span_qualifier_scorer


BINDING_GETTERS = keydefaultdict(make_binding_getter)
BINDING_SETTERS = keydefaultdict(make_binding_setter)


# noinspection PyMethodOverriding
class TrainableSpanQualifier(TrainablePipe):
    """Create a generic span classification component"""

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        candidate_getter: Callable[
            [Doc], Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]
        ],
        name: str = "span_qualifier",
        scorer: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        vocab: Vocab
            Spacy vocabulary
        model: Model
            The model to extract the spans
        name: str
            Name of the component
        candidate_getter: Callable[[Doc], Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]]
            Method to call to extract the candidate spans and the qualifiers
            to predict or train on.
        scorer: Optional[Callable]
            Method to call to score predictions
        """  # noqa: E501

        super().__init__(vocab, model, name)

        self.cfg["qualifiers"]: Optional[Tuple[str]] = ()
        self.candidate_getter = candidate_getter

        self.bindings: List[Binding] = []
        self.ner_labels_indices: Optional[Dict[str, int]] = None

        if scorer is None:
            self.scorer = make_span_qualifier_scorer(candidate_getter)
        else:
            self.scorer = scorer

    def to_disk(self, path, *, exclude=tuple()):
        # This will receive the directory path + /my_component
        super().to_disk(path, exclude=exclude)
        data_path = path / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "bindings": self.bindings,
                },
                f,
            )

    def from_disk(self, path, exclude=tuple()):
        super().from_disk(path, exclude=exclude)
        # This will receive the directory path + /my_component
        data_path = path / "data.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.bindings = data["bindings"]
        return self

    @property
    def qualifiers(self) -> Tuple[str]:
        """Return the qualifiers predicted by the component"""
        return self.cfg["qualifiers"]

    @property
    def labels(self) -> List[str]:
        return ["{}={}".format(a, b) for a, b in self.bindings]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        raise Exception("Cannot add a new label to the pipe")

    def predict(
        self, docs: List[Doc]
    ) -> Tuple[
        Dict[str, Ints2d],
        Spans,
        List[Optional[Spans]],
        List[SpanGroups],
        List[List[str]],
    ]:
        """
        Apply the pipeline's model to a batch of docs, without modifying them.

        Parameters
        ----------
        docs: List[Doc]

        Returns
        -------
        # noqa: E501
        Tuple[Dict[str, Ints2d], Spans, List[Spans], List[SpanGroups], List[List[str]]]
            The predicted list of 1-hot label sequence as a tensor
            that represent the labels of spans for all the batch,
            the list of all spans, and the span groups and ents in case the "label_"
            qualifier is updated
        """
        spans, ents, span_groups, spans_qlf, spans_array = self._get_span_data(docs)

        return (
            self.model.predict(
                (
                    docs,
                    self.model.ops.asarray(spans_array),
                    None,
                    True,
                )
            )[1],
            spans,
            ents,
            span_groups,
            spans_qlf,
        )

    def set_annotations(
        self,
        docs: List[Doc],
        predictions: Tuple[
            Dict[str, Ints2d],
            Spans,
            List[Optional[Spans]],
            List[SpanGroups],
            List[List[str]],
        ],
        **kwargs,
    ) -> None:
        """
        Modify the spans of a batch of `spacy.tokens.Span` objects, using the
        predicted labels.

        # noqa: E501
        Parameters
        ----------
        docs: List[Doc]
            The docs to update, not used in this function
        predictions: Tuple[Dict[str, Ints2d], Spans, List[SpanGroups], List[Optional[Spans]]]
            Tuple returned by the `predict` method, containing:
            - the label predictions. This is a 2d boolean tensor of shape
              (`batch_size`, `len(self.bindings)`)
            - the spans to update
            - the ents to reassign if the "label_" qualifier is updated
            - the span groups dicts to reassign if the "label_" qualifier is updated
            - the qualifiers for each span
        """
        output, spans, ents, span_groups, spans_qlf = predictions
        one_hot = output["labels"]
        for span, span_one_hot, span_qualifiers in zip(spans, one_hot, spans_qlf):
            for binding, is_present in zip(self.bindings, span_one_hot):
                if is_present and binding[0] in span_qualifiers:
                    BINDING_SETTERS[binding](span)

        # Because of the specific nature of the ".label_" attribute, we need to
        # reassign the ents on `doc.ents` (if `span_getter.from_ents`) and the spans
        # groups mentioned in `span_getter.from_spans_groups` on `doc.spans`
        if "label_" in self.qualifiers or "label" in self.qualifiers:
            if ents is not None:
                for doc, doc_ents in zip(docs, ents):
                    if doc_ents is not None:
                        doc.ents = doc_ents
            if span_groups is not None:
                for doc, doc_span_groups in zip(docs, span_groups):
                    doc.spans.update(doc_span_groups)

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to begin_update and get_loss.

        Unlike standard TrainablePipe components, the discrete ops (best selection
        of labels) is performed by the model directly (`begin_update` returns the loss
        and the predictions)

        Parameters
        ----------
        examples: Iterable[Example]
        drop: float = 0.0

        set_annotations: bool
            Whether to update the document with predicted spans
        sgd: Optional[Optimizer]
            Optimizer
        losses: Optional[Dict[str, float]]
            Dict of loss, updated in place

        Returns
        -------
        Dict[str, float]
            Updated losses dict
        """

        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)
        examples = list(examples)

        # run the model
        docs = [eg.predicted for eg in examples]
        (
            spans,
            ents,
            span_groups,
            spans_qlf,
            spans_array,
            targets,
        ) = self.examples_to_truth(examples)
        (loss, predictions), backprop = self.model.begin_update(
            (docs, spans_array, targets, set_annotations)
        )
        loss, gradient = self.get_loss(examples, loss)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        if set_annotations:
            self.set_annotations(
                spans,
                (
                    predictions,
                    spans,
                    ents,
                    span_groups,
                    spans_qlf,
                ),
            )

        losses[self.name] = loss

        return loss

    def get_loss(self, examples: Iterable[Example], loss) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        return float(loss.item()), self.model.ops.xp.array([1])

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize the pipe for training, using a representative set
        of data examples.

        Gather the qualifier values by iterating on the spans and their qualifiers
        matching the rules defined in the `candidate_getter`, and retrieving the
        values of the qualifiers.

        Parameters
        ----------
        get_examples: Callable[[], Iterable[Example]]
            Method to sample some examples
        nlp: spacy.Language
            Unused spacy model
        labels
            Unused list of labels
        """
        qualifier_values = defaultdict(set)
        for eg in get_examples():
            spans, *_, spans_qualifiers = self.candidate_getter(eg.reference)
            for span, span_qualifiers in zip(spans, spans_qualifiers):
                for qualifier in span_qualifiers:
                    value = BINDING_GETTERS[qualifier](span)
                    qualifier_values[qualifier].add(value)

        qualifier_values = {
            key: sorted(values, key=str) for key, values in qualifier_values.items()
        }

        self.cfg["qualifiers"] = sorted(qualifier_values.keys())
        # groups:
        #   num binding_groups (e.g. ["events", "negation"])
        # * num label combinations in this group
        # * positive labels in this combination
        self.cfg["groups"] = [
            [((key, value),) for value in sorted(values, key=str)]
            for key, values in qualifier_values.items()
        ]
        groups_bindings = [
            list(
                dict.fromkeys(
                    [
                        binding
                        for combination_bindings in group_combinations
                        for binding in combination_bindings
                    ]
                )
            )
            for group_combinations in self.cfg["groups"]
        ]
        self.bindings = [
            binding for group_bindings in groups_bindings for binding in group_bindings
        ]
        self.model.attrs["set_n_labels"](len(self.bindings))

        # combinations_one_hot: list of bool arrays of shape
        #   num binding_groups (e.g. ["events", "negation"])
        # * num bindings in this group (eg ["start", "stop"], [True, False])
        combinations_one_hot: List[List[List[bool]]] = [
            [
                [binding in combination_bindings for binding in group_bindings]
                for combination_bindings in group_combinations
            ]
            for group_combinations, group_bindings in zip(
                self.cfg["groups"], groups_bindings
            )
        ]
        # groups_indices:
        #   num binding_groups (e.g. ["events", "negation"])
        # * num label combinations in this group
        # * presence or absence (bool) of the bindings of this groups in the combination
        groups_bindings_indices = [
            [self.bindings.index(binding) for binding in group_bindings]
            for group_bindings in groups_bindings
        ]

        self.model.attrs["set_label_groups"](
            combinations_one_hot,
            groups_bindings_indices,
        )

        # Neural network initialization
        sub_batch = list(islice(get_examples(), NUM_INITIALIZATION_EXAMPLES))
        doc_sample = [eg.reference for eg in sub_batch]
        spans, *_, spans_array, targets = self.examples_to_truth(sub_batch)
        if len(spans) == 0:
            raise ValueError(
                "Call begin_training with relevant entities "
                "and relations annotated in "
                "at least a few reference examples!"
            )

        self.model.initialize(X=doc_sample, Y=spans_array)

    def _get_span_data(
        self, docs: List[Doc]
    ) -> Tuple[
        Spans,
        List[Optional[Spans]],
        List[SpanGroups],
        List[List[str]],
        np.ndarray,
    ]:
        spans = []
        ents, span_groups = [], []
        spans_qualifiers = []
        for doc_idx, doc in enumerate(docs):
            doc_spans, doc_ents, doc_span_groups, qlf = self.candidate_getter(doc)
            ents.append(doc_ents)
            span_groups.append(doc_span_groups)
            spans_qualifiers.extend(qlf)
            spans.extend([(doc_idx, span) for span in doc_spans])
        spans = list(spans)
        spans_array = np.zeros((len(spans), 3), dtype=int)
        for i, (doc_idx, span) in enumerate(spans):
            spans_array[i] = (
                doc_idx,
                span.start,
                span.end,
            )

        return (
            [span for i, span in spans],
            ents,
            span_groups,
            spans_qualifiers,
            spans_array,
        )

    def examples_to_truth(
        self, examples: List[Example]
    ) -> Tuple[
        Spans,
        List[Spans],
        List[SpanGroups],
        List[List[str]],
        Ints2d,
        List[Ints2d],
    ]:
        """

        Converts the spans of the examples into a list
        of (doc_idx, label_idx, begin, end) tuple as a tensor,
        and the labels of the spans into a list of 1-hot label sequence

        Parameters
        ----------
        examples: List[Example]

        Returns
        -------
        Tuple[Spans,List[Spans],List[SpanGroups],List[List[str]],Ints2d,List[Ints2d]]
            The list of spans, the spans tensor, the qualifiers tensor, and the
            list of entities and span groups to reassign them if the label_ attribute
            is part of the updated qualifiers
        """  # noqa E501
        spans, ents, span_groups, spans_qualifiers, spans_array = self._get_span_data(
            [eg.reference for eg in examples]
        )
        targets = [
            np.zeros((len(spans), len(group_combinations)), dtype=int)
            for group_combinations in self.cfg["groups"]
        ]
        for span_idx, span in enumerate(spans):
            span_bindings = []
            for j, binding in enumerate(self.bindings):
                if binding[0] in spans_qualifiers[span_idx] and BINDING_GETTERS[
                    binding
                ](span):
                    span_bindings.append(binding)
            for group_idx, group in enumerate(self.cfg["groups"]):
                for comb_idx, group_combination in enumerate(group):
                    if set(group_combination).issubset(set(span_bindings)):
                        targets[group_idx][span_idx, comb_idx] = 1

        return (
            spans,
            ents,
            span_groups,
            spans_qualifiers,
            self.model.ops.asarray(spans_array),
            [self.model.ops.asarray(arr) for arr in targets],
        )
