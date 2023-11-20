from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Sequence, Union

from rich.text import Span
from spacy.tokens import Doc

from edsnlp import registry
from edsnlp.utils.filter import filter_spans

SeqStr = Union[str, Sequence[str]]
SpanFilter = Union[bool, SeqStr]

SpanSetterMapping = Dict[str, SpanFilter]
SpanGetterMapping = Dict[str, SpanFilter]

SpanGetter = Union[
    SpanGetterMapping,
    Callable[[Doc], Iterable[Span]],
]
SpanSetter = Union[
    SpanSetterMapping,
    Callable[[Doc, Iterable[Span]], Any],
]


def get_spans(doc, span_getter):
    if span_getter is None:
        yield doc[:]
        return
    if callable(span_getter):
        yield from span_getter(doc)
        return
    for key, span_filter in span_getter.items():
        if key == "*":
            candidates = (span for group in doc.spans.values() for span in group)
        else:
            candidates = doc.spans.get(key, ()) if key != "ents" else doc.ents
        if span_filter is True:
            yield from candidates
        else:
            for span in candidates:
                if span.label_ in span_filter:
                    yield span


def get_spans_with_group(doc, span_getter):
    if span_getter is None:
        yield doc[:], None
        return
    if callable(span_getter):
        yield from span_getter(doc)
        return
    for key, span_filter in span_getter.items():
        if key == "*":
            candidates = (
                (span, group) for group in doc.spans.values() for span in group
            )
        else:
            candidates = doc.spans.get(key, ()) if key != "ents" else doc.ents
            candidates = ((span, key) for span in candidates)
        if span_filter is True:
            yield from candidates
        else:
            for span, group in candidates:
                if span.label_ in span_filter:
                    yield span, group


def set_spans(doc, matches, span_setter):
    if callable(span_setter):
        span_setter(doc, matches)
    else:
        match_all = []
        label_to_group = defaultdict(list)
        for name, spans_filter in span_setter.items():
            if name != "ents" and name != "*":
                doc.spans.setdefault(name, [])
            if spans_filter:
                if spans_filter is True:
                    match_all.append(name)
                else:
                    for label in spans_filter:
                        label_to_group[label].append(name)

        new_ents = [] if "ents" in span_setter else None

        for span in matches:
            for group in match_all + label_to_group[span.label_]:
                if group == "ents":
                    new_ents.append(span)
                elif group == "*":
                    doc.spans.setdefault(span.label_, []).append(span)
                else:
                    doc.spans[group].append(span)
        if new_ents is not None:
            doc.ents = filter_spans((*new_ents, *doc.ents))
    return doc


def validate_span_setter(value: Union[SeqStr, Dict[str, SpanFilter]]) -> SpanSetter:
    if callable(value):
        return value
    if isinstance(value, str):
        return {value: True}
    if isinstance(value, list):
        return {group: True for group in value}
    elif isinstance(value, dict):
        new_value = {}
        for k, v in value.items():
            if isinstance(v, bool):
                new_value[k] = v
            elif isinstance(v, str):
                new_value[k] = [v]
            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                new_value[k] = v
            else:
                raise TypeError(
                    f"Invalid entry {value} ({type(value)}) for SpanSetterArg, "
                    f"expected bool/string(s), dict of bool/string(s) or callable"
                )
        return new_value
    else:
        raise TypeError(
            f"Invalid entry {value} ({type(value)}) for SpanSetterArg, "
            f"expected bool/string(s), dict of bool/string(s) or callable"
        )


def validate_span_getter(
    value: Union[SeqStr, Dict[str, SpanFilter]], optional: bool = False
) -> SpanSetter:
    if value is None:
        if optional:
            return None
        raise ValueError(
            "Mising entry for SpanGetterArg, "
            "expected bool/string(s), dict of bool/string(s) or callable"
        )
    if callable(value):
        return value
    if isinstance(value, str):
        return {value: True}
    if isinstance(value, list):
        return {group: True for group in value}
    elif isinstance(value, dict):
        new_value = {}
        for k, v in value.items():
            if isinstance(v, bool):
                new_value[k] = v
            elif isinstance(v, str):
                new_value[k] = [v]
            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                new_value[k] = v
            else:
                raise TypeError(
                    f"Invalid entry {value} ({type(value)}) for SpanGetterArg, "
                    f"expected bool/string(s), dict of bool/string(s) or callable"
                )
        return new_value
    else:
        raise TypeError(
            f"Invalid entry {value} ({type(value)}) for SpanGetterArg, "
            f"expected bool/string(s), dict of bool/string(s) or callable"
        )


class SpanSetterArg:
    """
    Valid values for the `span_setter` argument of a component can be :

    - a (doc, matches) -> None callable
    - a span group name
    - a list of span group names
    - a dict of group name to True or list of labels

    The group name `"ents"` is a special case, and will add the matches to `doc.ents`

    Examples
    --------
    - `span_setter=["ents", "ckd"]` will add the matches to both `doc.ents` and
    `doc.spans["ckd"]`. It is equivalent to `{"ents": True, "ckd": True}`.
    - `span_setter={"ents": ["foo", "bar"]}` will add the matches with label
    "foo" and "bar" to `doc.ents`.
    - `span_setter="ents"` will add all matches only to `doc.ents`.
    - `span_setter="ckd"` will add all matches only to `doc.spans["ckd"]`.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None) -> SpanSetter:
        return validate_span_setter(value)


class SpanGetterArg:
    """
    Valid values for the `span_getter` argument of a component can be :

    - a (doc) -> spans callable
    - a span group name
    - a list of span group names
    - a dict of group name to True or list of labels

    The group name `"ents"` is a special case, and will get the matches from `doc.ents`

    Examples
    --------
    - `span_getter=["ents", "ckd"]` will get the matches from both `doc.ents` and
    `doc.spans["ckd"]`. It is equivalent to `{"ents": True, "ckd": True}`.
    - `span_getter={"ents": ["foo", "bar"]}` will get the matches with label
    "foo" and "bar" from `doc.ents`.
    - `span_getter="ents"` will get all matches from `doc.ents`.
    - `span_getter="ckd"` will get all matches from `doc.spans["ckd"]`.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None) -> SpanSetter:
        return validate_span_setter(value)


if TYPE_CHECKING:
    SpanGetterArg = Union[  # noqa: F811
        str,
        Sequence[str],
        SpanGetterMapping,
        Callable[[Doc], Iterable[Span]],
    ]
    SpanSetterArg = Union[  # noqa: F811
        str,
        Sequence[str],
        SpanSetterMapping,
        Callable[[Doc, Iterable[Span]], Any],
    ]


@registry.misc.register("eds.span_sentence_getter")
class make_span_sentence_getter:
    def __init__(
        self,
        span_getter: SpanGetterArg,
        min_context_words: int = 0,
    ):
        self.min_context_words = min_context_words
        self.span_getter = span_getter

    def __call__(self, doc: Doc):
        ctx = self.min_context_words
        spans = (
            doc[min(e[0].sent.start, e.start - ctx) : max(e[-1].sent.end, e.end + ctx)]
            for e in get_spans(doc, self.span_getter)
        )

        return filter_spans(spans)
