from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from rich.text import Span
from spacy.tokens import Doc
from typing_extensions import NotRequired, TypedDict

from edsnlp.utils.span_getters import SpanGetterArg, get_spans

Binding = Tuple[str, Any]


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
    binding: Binding
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


BINDING_GETTERS = keydefaultdict(make_binding_getter)
BINDING_SETTERS = keydefaultdict(make_binding_setter)

BindingCandidateGetter = TypedDict(
    "BindingCandidateGetter",
    {
        "span_getter": SpanGetterArg,
        "qualifiers": NotRequired[Sequence[str]],
        "label_constraints": NotRequired[Dict[str, List[str]]],
    },
)

BindingCandidateGetterArg = Union[
    BindingCandidateGetter,
    Callable[[Doc], Tuple[List[Span], List[List[Any]]]],
]
"""
Either a dict with the keys described below, or a function that takes a `Doc` and
returns a tuple of (spans, list of multiple attributes per span). One of `qualifiers`
or `label_constraints` must be given.

Parameters
----------
span_getter: SpanGetterMapping
    Whether to look into `doc.ents` for spans to classify. If a list of strings
qualifiers: Optional[Sequence[str]]
    The qualifiers to predict or train on. If None, keys from the
    `label_constraints` will be used
label_constraints: Optional[Dict[str, List[str]]]
    Constraints to select qualifiers for each span depending on their labels.
    Keys of the dict are the qualifiers and values are the labels for which
    the qualifier is allowed. If None, all qualifiers will be used for all spans
"""


def get_candidates(doc: Doc, candidate_getter: BindingCandidateGetterArg):
    """
    Make a span qualifier candidate getter function.

    Parameters
    ----------
    doc: Doc
        Document to get the candidates from
    candidate_getter: BindingCandidateGetterArg
        Options on how to get the qualification candidates
        (span x attribute paths)

    Returns
    -------
    Tuple[List[Span], List[List[Any]]]
    """

    if callable(candidate_getter):
        return candidate_getter(doc)

    qualifiers = candidate_getter.get("qualifiers")
    label_constraints = candidate_getter.get("label_constraints")

    if qualifiers is None and label_constraints is None:
        raise ValueError(
            "Either `qualifiers` or `label_constraints` must be given to "
            "provide the qualifiers to predict / train on."
        )
    elif qualifiers is None:
        qualifiers = list(label_constraints.keys())

    label_constraints = label_constraints
    qualifiers = qualifiers

    flattened_spans = list(get_spans(doc, candidate_getter["span_getter"]))

    if label_constraints:
        span_qualifiers = [
            [
                qualifier
                for qualifier in qualifiers
                if qualifier not in label_constraints
                or span.label_ in label_constraints[qualifier]
            ]
            for span in flattened_spans
        ]
    else:
        span_qualifiers = [qualifiers] * len(flattened_spans)
    return flattened_spans, span_qualifiers
