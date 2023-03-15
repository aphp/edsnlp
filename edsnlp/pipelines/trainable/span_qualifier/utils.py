from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from spacy import registry
from spacy.tokens import Doc, Span

Binding = Tuple[str, Any]
Spans = List[Span]
SpanGroups = Dict[str, Spans]


@registry.misc("eds.candidate_span_qualifier_getter")
class make_candidate_getter:
    def __init__(
        self,
        from_ents: Optional[Union[bool, Sequence[str]]] = None,
        from_span_groups: Union[
            bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
        ] = False,
        qualifiers: Optional[Sequence[str]] = None,
        label_constraints: Optional[Dict[str, List[str]]] = None,
    ):

        """
        Make a span qualifier candidate getter function.

        Parameters
        ----------
        from_ents: Union[bool, Sequence[str]]
            Whether to look into `doc.ents` for spans to classify. If a list of strings
            is provided, only the span of the given labels will be considered. If None
            and `from_span_groups` is False, labels mentioned in `label_constraints`
            will be used.
        from_span_groups: Union[bool, Sequence[str], Mapping[str, Sequence[str]]]
            Whether to look into `doc.spans` for spans to classify:

            - If True, all span groups will be considered
            - If False, no span group will be considered
            - If a list of str is provided, only these span groups will be kept
            - If a mapping is provided, the keys are the span group names and the values
              are either a list of allowed labels in the group or True to keep them all
        qualifiers: Optional[Sequence[str]]
            The qualifiers to predict or train on. If None, keys from the
            `label_constraints` will be used
        label_constraints: Optional[Dict[str, List[str]]]
            Constraints to select qualifiers for each span depending on their labels.
            Keys of the dict are the qualifiers and values are the labels for which
            the qualifier is allowed. If None, all qualifiers will be used for all spans

        Returns
        -------
        Callable[[Doc], Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]]
        """

        if qualifiers is None and label_constraints is None:
            raise ValueError(
                "Either `qualifiers` or `label_constraints` must be given to "
                "provide the qualifiers to predict / train on."
            )
        elif qualifiers is None:
            qualifiers = list(label_constraints.keys())

        if not from_span_groups and from_ents is None:
            if label_constraints is None:
                from_ents = True
            else:
                from_ents = sorted(
                    set(
                        label
                        for qualifier in label_constraints
                        for label in label_constraints[qualifier]
                    )
                )

        self.from_ents = from_ents
        self.from_span_groups = from_span_groups
        self.label_constraints = label_constraints
        self.qualifiers = qualifiers

    def __call__(
        self,
        doc: Doc,
    ) -> Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]:
        flattened_spans = []
        span_groups = {}
        ents = None
        if self.from_ents:
            # /!\ doc.ents is not a list but a Span iterator, so to ensure referential
            # equality between the spans of `flattened_spans` and `ents`,
            # we need to convert it to a list to "extract" the spans first
            ents = list(doc.ents)
            if isinstance(self.from_ents, Sequence):
                flattened_spans.extend(
                    span for span in ents if span.label_ in self.from_ents
                )
            else:
                flattened_spans.extend(ents)

        if self.from_span_groups:
            if isinstance(self.from_span_groups, Mapping):
                for name, labels in self.from_span_groups.items():
                    if labels:
                        span_groups[name] = list(doc.spans.get(name, ()))
                        if isinstance(labels, Sequence):
                            flattened_spans.extend(
                                span
                                for span in span_groups[name]
                                if span.label_ in labels
                            )
                        else:
                            flattened_spans.extend(span_groups[name])
            elif isinstance(self.from_span_groups, Sequence):
                for name in self.from_span_groups:
                    span_groups[name] = list(doc.spans.get(name, ()))
                    flattened_spans.extend(span_groups[name])
            else:
                for name, spans_ in doc.spans.items():
                    # /!\ spans_ is not a list but a SpanGroup, so to ensure referential
                    # equality between the spans of `flattened_spans` and `span_groups`,
                    # we need to convert it to a list to "extract" the spans first
                    span_groups[name] = list(spans_)
                    flattened_spans.extend(span_groups[name])

        if self.label_constraints:
            span_qualifiers = [
                [
                    qualifier
                    for qualifier in self.qualifiers
                    if qualifier not in self.label_constraints
                    or span.label_ in self.label_constraints[qualifier]
                ]
                for span in flattened_spans
            ]
        else:
            span_qualifiers = [self.qualifiers] * len(flattened_spans)
        return flattened_spans, ents, span_groups, span_qualifiers


def _check_path(path: str):
    assert [letter.isalnum() or letter == "_" or letter == "." for letter in path], (
        "The label must be a path of valid python identifier to be used as a getter"
        "in the following template: span.[YOUR_LABEL], such as `label_` or `_.negated"
    )


def make_binding_getter(qualifier: Union[str, Binding]):
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


def make_binding_setter(binding: Binding):
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
