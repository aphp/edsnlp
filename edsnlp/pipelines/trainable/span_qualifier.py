import pickle
from collections import defaultdict
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import spacy
from spacy import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.vocab import Vocab
from thinc.api import Model, Optimizer
from thinc.backends import NumpyOps
from thinc.config import Config
from thinc.model import set_dropout_rate
from thinc.types import Ints2d
from wasabi import Printer

msg = Printer()

Binding = Tuple[str, Any]


NUM_INITIALIZATION_EXAMPLES = 10


span_qualifier_default_config = """
[model]
    @architectures = "eds.span_multi_classifier.v1"
    projection_mode = "dot"
    pooler_mode = "max"

    [model.tok2vec]
        @architectures = "spacy.Tok2Vec.v1"

    [model.tok2vec.embed]
        @architectures = "spacy.MultiHashEmbed.v1"
        width = 96
        rows = [5000, 2000, 1000, 1000]
        attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
        include_static_vectors = false

    [model.tok2vec.encode]
        @architectures = "spacy.MaxoutWindowEncoder.v1"
        width = ${model.tok2vec.embed.width}
        window_size = 1
        maxout_pieces = 3
        depth = 4

[scorer]
    @scorers = "eds.span_qualifier_scorer.v1"
"""

SPAN_CLASSIFIER_DEFAULTS = Config().from_str(span_qualifier_default_config)
np_ops = NumpyOps()


@Language.factory(
    "span_qualifier",
    default_config=SPAN_CLASSIFIER_DEFAULTS,
    requires=["doc.ents", "doc.spans"],
    assigns=["doc.ents", "doc.spans"],
    default_score_weights={
        "accuracy": 1.0,
    },
)
def create_component(
    nlp: Language,
    model: Model,
    qualifiers: Sequence[str],
    name: str = "span_qualifier",
    ner_constraints: Optional[Mapping[str, Sequence[str]]] = None,
    on_ents: Union[bool, Sequence[str]] = True,
    on_span_groups: Union[
        bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
    ] = True,  # noqa: E501
    scorer: Optional[Callable] = None,
):
    """Construct a TrainableQualifier component."""
    return TrainableSpanClassifier(
        vocab=nlp.vocab,
        model=model,
        name=name,
        qualifiers=qualifiers,
        ner_constraints=ner_constraints,
        on_ents=on_ents,
        on_span_groups=on_span_groups,
        scorer=scorer,
    )


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
    ner_constraints = cfg["ner_constraints"]
    for eg_idx, eg in enumerate(examples):
        for span in get_spans(eg.predicted, cfg["on_ents"], cfg["on_span_groups"])[0]:
            for qualifier in cfg["qualifiers"]:
                if (
                    ner_constraints is None
                    or qualifier not in ner_constraints
                    or span.label_ in ner_constraints[qualifier]
                ):
                    labels[qualifier][0].append(QUALIFIER_GETTERS[qualifier](span))
                    labels["__all__"][0].append(QUALIFIER_GETTERS[qualifier](span))

        for span in get_spans(eg.reference, cfg["on_ents"], cfg["on_span_groups"])[0]:
            for qualifier in cfg["qualifiers"]:
                if (
                    ner_constraints is None
                    or qualifier not in ner_constraints
                    or span.label_ in ner_constraints[qualifier]
                ):
                    labels[qualifier][1].append(QUALIFIER_GETTERS[qualifier](span))
                    labels["__all__"][1].append(QUALIFIER_GETTERS[qualifier](span))

    results = {
        name: sum(p == g for p, g in zip(pred, gold)) / len(pred)
        for name, (pred, gold) in labels.items()
    }
    return {"accuracy": results["__all__"]}


@spacy.registry.scorers("eds.span_qualifier_scorer.v1")
def make_span_qualifier_scorer():
    return span_qualifier_scorer


def get_spans(
    doc: Doc,
    on_ents: Union[bool, Sequence[str]] = True,
    on_span_groups: Union[
        bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
    ] = True,  # noqa: E501
) -> Tuple[List[Span], Dict[str, List[Span]], Optional[List[Span]]]:
    """
    Get the spans from the doc

    Parameters
    ----------
    doc: Doc
        The document to extract the spans from
    on_ents: Union[bool, Sequence[str]]
        Whether to extract the spans from the entities or not
    on_span_groups: Union[bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]]
        Whether to extract the spans from the span groups or not

    Returns
    -------
    Tuple[List[Span], Dict[str, List[Span]], Optional[List[Span]]]
        The extracted spans
    """
    flattened_spans = []
    span_groups = {}
    ents = None
    if on_ents:
        # /!\ doc.ents is not a list but a Span iterator, so to ensure referential
        # equality between the spans of `flattened_spans` and `ents`,
        # we need to convert it to a list to "extract" the spans first
        ents = list(doc.ents)
        if isinstance(on_ents, Sequence):
            flattened_spans.extend(span for span in ents if span.label_ in on_ents)
        else:
            flattened_spans.extend(ents)

    if on_span_groups:
        if isinstance(on_span_groups, Mapping):
            for name, labels in on_span_groups.items():
                if labels:
                    span_groups[name] = list(doc.spans.get(name, ()))
                    if isinstance(labels, Sequence):
                        flattened_spans.extend(
                            span for span in span_groups[name] if span.label_ in labels
                        )
                    else:
                        flattened_spans.extend(span_groups[name])
        else:
            for name, spans_ in doc.spans.items():
                # /!\ spans_ is not a list but a SpanGroup, so to ensure referential
                # equality between the spans of `flattened_spans` and `span_groups`,
                # we need to convert it to a list to "extract" the spans first
                span_groups[name] = list(spans_)
                flattened_spans.extend(span_groups[name])

    return flattened_spans, span_groups, ents


def _check_path(path: str):
    assert [letter.isalnum() or letter == "_" or letter == "." for letter in path], (
        "The label must be a path of valid python identifier to be used as a getter"
        "in the following template: span.[YOUR_LABEL], such as `label_` or `_.negated"
    )


def make_qualifier_getter(qualifier: Union[str, Binding]):
    """
    Make a qualifier getter

    Parameters
    ----------
    qualifier: Union[str, Binding]
        Either one of the following:
        - a path to a nested attributes of the span, such as "qualifier_" or "_.negated"
        - a tuple of (key, value) equality, such as `("_.date.mode", "PASSED")`

    Returns
    -------
    Callable[[Span], bool]
        The qualifier getter
    """
    if isinstance(qualifier, tuple):
        path, value = qualifier
        _check_path(path)
        return eval(f"lambda span: span.{path} == value", {"value": value}, {})
    else:
        _check_path(qualifier)
        return eval(f"lambda span: span.{qualifier}")


def make_qualifier_setter(binding: Binding):
    """
    Make a qualifier setter

    Parameters
    ----------
    qualifier: Binding
        A pair of
        - a path to a nested attributes of the span, such as `qualifier_` or `_.negated`
        - a value assignment

    Returns
    -------
    Callable[[Span]]
        The qualifier setter
    """
    path, value = binding
    _check_path(path)
    fn_string = f"""def fn(span): span.{path} = value"""
    loc = {"value": value}
    exec(fn_string, loc, loc)
    return loc["fn"]


K = TypeVar("K")
V = TypeVar("V")


class keydefaultdict(dict):
    def __init__(self, default_factory: Callable[[K], V]):
        super().__init__()
        self.default_factory = default_factory

    def __missing__(self, key: K) -> V:
        ret = self[key] = self.default_factory(key)
        return ret


QUALIFIER_GETTERS = keydefaultdict(make_qualifier_getter)
QUALIFIER_SETTERS = keydefaultdict(make_qualifier_setter)


# noinspection PyMethodOverriding
class TrainableSpanClassifier(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        qualifiers: Sequence[str],
        name: str = "span_qualifier",
        ner_constraints: Optional[Mapping[str, Sequence[str]]] = None,
        on_ents: Union[bool, Sequence[str]] = True,
        on_span_groups: Union[
            bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
        ] = True,  # noqa: E501
        scorer: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a general span classification component

        Parameters
        ----------
        vocab: Vocab
            Spacy vocabulary
        model: Model
            The model to extract the spans
        name: str
            Name of the component
        qualifiers: Sequence[str]
            List of paths to nested attributes to predict in the spans
            such as `label_` or `_.negated`
        ner_constraints: Optional[Mapping[str, Sequence[str]]]
            Mapping from qualifier to a list of ner labels for this qualifier.
            Useful to only predict certain qualifiers for certain spans of a given type.
        on_ents: Union[bool, Sequence[str]]
            Whether to look into `doc.ents` for spans to classify. If a list of strings
            is provided, only the span of the given labels will be considered.
        on_span_groups: Union[bool, Sequence[str], Mapping[str, Sequence[str]]]
            Whether to look into `doc.spans` for spans to classify:

            - If True, all span groups will be considered
            - If False, no span group will be considered
            - If a list of str is provided, only these span groups will be kept
            - If a mapping is provided, the keys are the span group names and the values
              are either a list of allowed labels in the group or True to keep them all
        scorer: Optional[Callable]
            Method to call to score predictions
        """

        super().__init__(vocab, model, name)

        self.cfg["qualifiers"]: Optional[Tuple[str]] = tuple(qualifiers)
        self.cfg["ner_constraints"] = ner_constraints
        self.cfg["on_ents"] = on_ents
        self.cfg["on_span_groups"] = on_span_groups

        self.bindings: List[Binding] = []
        self.ner_labels_indices: Optional[Dict[str, int]] = None

        self.scorer = scorer

    def to_disk(self, path, *, exclude=tuple()):
        # This will receive the directory path + /my_component
        super().to_disk(path, exclude=exclude)
        data_path = path / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "bindings": self.bindings,
                    "ner_labels_indices": self.ner_labels_indices,
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
        self.ner_labels_indices = data["ner_labels_indices"]
        return self

    @property
    def qualifiers(self) -> Tuple[str]:
        """Return the qualifiers predicted by the component"""
        return self.cfg["qualifiers"]

    @property
    def labels(self) -> List[str]:
        return self.bindings

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        raise Exception("Cannot add a new label to the pipe")

    def predict(
        self, docs: List[Doc]
    ) -> Tuple[
        Dict[str, Ints2d],
        List[Span],
        List[Dict[str, List[Span]]],
        List[Optional[List[Span]]],
    ]:
        """
        Apply the pipeline's model to a batch of docs, without modifying them.

        Parameters
        ----------
        docs: List[Doc]

        Returns
        -------
        # noqa: E501
        Tuple[Dict[str, Ints2d], List[Span], List[Dict[str, List[Span]]], List[Optional[List[Span]]]]
            The predicted list of 1-hot label sequence as a tensor
            that represent the labels of spans for all the batch,
            the list of all spans, and the span groups and ents in case the "label_"
            qualifier is updated
        """
        spans, span_groups, ents, spans_array = self._get_span_data(docs)

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
            span_groups,
            ents,
        )

    def set_annotations(
        self,
        docs: List[Doc],
        predictions: Tuple[
            Dict[str, Ints2d],
            List[Span],
            List[Dict[str, List[Span]]],
            List[Optional[List[Span]]],
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
        predictions: Tuple[Dict[str, Ints2d], List[Span], List[Dict[str, List[Span]]], List[Optional[List[Span]]]]
            Tuple returned by the `predict` method, containing:
            - the label predictions. This is a 2d boolean tensor of shape
              (`batch_size`, `len(self.bindings)`)
            - the spans to update
            - the span groups dicts to reassign if the "label_" qualifier is updated
            - the ents to reassign if the "label_" qualifier is updated
        """
        one_hot = predictions[0]["labels"]
        spans = predictions[1]
        span_groups = predictions[2]
        ents = predictions[3]
        ner_constraints = self.cfg["ner_constraints"]
        for span, span_one_hot in zip(spans, one_hot):
            for binding, is_present in zip(self.bindings, span_one_hot):
                if is_present and (
                    ner_constraints is None
                    or binding[0] not in ner_constraints
                    or span.label_ in ner_constraints[binding[0]]
                ):
                    QUALIFIER_SETTERS[binding](span)

        # Because of the specific nature of the ".label_" attribute, we need to
        # reassign the ents on `doc.ents` (if `on_ents`) and the spans groups mentioned
        # in `on_spans_groups` on `doc.spans`
        if "label_" in self.qualifiers or "label" in self.qualifiers:
            if self.cfg["on_ents"]:
                for doc, doc_ents in zip(docs, ents):
                    doc.ents = doc_ents
            if self.cfg["on_span_groups"]:
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
            spans_array,
            qualifiers_array,
        ) = self.examples_to_truth(examples)
        (loss, predictions), backprop = self.model.begin_update(
            (docs, spans_array, qualifiers_array, set_annotations)
        )
        loss, gradient = self.get_loss(examples, loss)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        if set_annotations:
            self.set_annotations(spans, predictions)

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

        1. Gather the qualifier values by iterating on the spans matching the rules
        defined in the `on_ents` and `on_span_groups` parameters, and retrieving the
        values of the qualifiers.
        2. If `ner_constraints` is not None, use only the spans with a label matching
        the list of span labels in `ner_constraints` for the given qualifier.

        Parameters
        ----------
        get_examples: Callable[[], Iterable[Example]]
            Method to sample some examples
        nlp: spacy.Language
            Unused spacy model
        labels
            Unused list of labels
        """
        qualifier_values = {qlf: set() for qlf in self.cfg["qualifiers"]}
        ner_constraints = self.cfg["ner_constraints"]
        for eg in get_examples():
            doc_spans = get_spans(
                eg.reference,
                self.cfg["on_ents"],
                self.cfg["on_span_groups"],
            )[0]
            for span in doc_spans:
                for qualifier in self.cfg["qualifiers"]:
                    if (
                        ner_constraints is None
                        or qualifier not in ner_constraints
                        or span.label_ in ner_constraints[qualifier]
                    ):
                        value = QUALIFIER_GETTERS[qualifier](span)
                        qualifier_values[qualifier].add(value)

        # groups:
        #   num binding_groups (e.g. ["events", "negation"])
        # * num label combinations in this group
        # * positive labels in this combination
        groups = [
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
            for group_combinations in groups
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
            for group_combinations, group_bindings in zip(groups, groups_bindings)
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

        # Set span labels constraints as a list of boolean lists
        if "set_ner_constraints" in self.model.attrs and ner_constraints is not None:
            self.ner_labels_indices = {
                k: i
                for i, k in enumerate(
                    ["_"]
                    + [
                        label
                        for allowed_ner_labels in self.cfg["ner_constraints"].values()
                        for label in allowed_ner_labels
                    ]
                )
            }
            ner_constraints = np.ones(
                (len(self.ner_labels_indices), len(self.bindings)), dtype=bool
            )
            for qualifier, allowed_ner_labels in self.cfg["ner_constraints"].items():
                for value in qualifier_values[qualifier]:
                    ner_constraints[:, self.bindings.index((qualifier, value))] = False
                    ner_constraints[
                        [
                            self.ner_labels_indices[label]
                            for label in allowed_ner_labels
                        ],
                        self.bindings.index((qualifier, value)),
                    ] = True
            self.model.attrs["set_ner_constraints"](ner_constraints)

        # Neural network initialization
        sub_batch = list(islice(get_examples(), NUM_INITIALIZATION_EXAMPLES))
        doc_sample = [eg.reference for eg in sub_batch]
        spans, *_, spans_array, qualifiers_array = self.examples_to_truth(sub_batch)
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
        List[Span],
        List[Dict[str, List[Span]]],
        List[Optional[List[Span]]],
        np.ndarray,
    ]:
        spans = set()
        ents, span_groups = [], []
        for doc_idx, doc in enumerate(docs):
            doc_spans, doc_ents, doc_span_groups = get_spans(
                doc,
                self.cfg["on_ents"],
                self.cfg["on_span_groups"],
            )
            ents.append(doc_ents)
            span_groups.append(span_groups)
            spans.update([(doc_idx, span) for span in doc_spans])
        spans = list(spans)
        spans_array = np.zeros((len(spans), 4), dtype=int)
        for i, (doc_idx, span) in enumerate(spans):
            spans_array[i] = (
                doc_idx,
                self.ner_labels_indices.get(span.label_, 0)
                if self.ner_labels_indices is not None
                else 0,
                span.start,
                span.end,
            )

        return [span for i, span in spans], ents, span_groups, spans_array

    def examples_to_truth(
        self, examples: List[Example]
    ) -> Tuple[
        List[Span],
        List[Dict[str, List[Span]]],
        List[Optional[List[Span]]],
        Ints2d,
        Ints2d,
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
        # noqa: E501
        List[Dict[str, List[Span]]], List[Optional[List[Span]]]], Tuple[List[Span], Ints2d, Ints2d
            The list of spans, the spans tensor, the qualifiers tensor, and the
            list of entities and span groups to reassign them if the label_ attribute
            is part of the updated qualifiers
        """
        spans, ents, span_groups, spans_array = self._get_span_data(
            [eg.reference for eg in examples]
        )
        ner_constraints = self.cfg["ner_constraints"]
        binding_array = np.zeros((len(spans), len(self.bindings)), dtype=int)
        for i, span in enumerate(spans):
            for j, binding in enumerate(self.bindings):
                if (
                    ner_constraints is None
                    or binding[0] not in ner_constraints
                    or span.label_ in ner_constraints[binding[0]]
                ):
                    binding_array[i, j] = QUALIFIER_GETTERS[binding](span)
        return (
            spans,
            ents,
            span_groups,
            self.model.ops.asarray(spans_array),
            self.model.ops.asarray(binding_array),
        )
