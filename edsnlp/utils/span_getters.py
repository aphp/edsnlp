from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

from pydantic import NonNegativeInt
from spacy.tokens import Doc, Span

from edsnlp import registry
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.typing import AsList

SeqStr = AsList[str]
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


@registry.misc.register("eds.span_context_getter")
class make_span_context_getter:
    """
    Create a span context getter.

    Parameters
    ----------
    context_words : NonNegativeInt
        Minimum number of words to include on each side of the span.
    context_sents : Optional[NonNegativeInt]
        Minimum number of sentences to include on each side of the span:

        - 0: don't use sentences to build the context.
        - 1: include the sentence of the span.
        - n: include n sentences on each side of the span.

        By default, 0 if the document has no sentence annotations, 1 otherwise.
    """

    def __init__(
        self,
        context_words: NonNegativeInt = 0,
        context_sents: Optional[NonNegativeInt] = 1,
        span_getter: Optional[SpanGetterArg] = None,
    ):
        self.context_words = context_words
        self.context_sents = context_sents
        self.span_getter = validate_span_getter(span_getter, optional=True)

    def __call__(self, span: Union[Doc, Span]) -> Union[Span, List[Span]]:
        if isinstance(span, Doc):  # pragma: no cover
            return [self(s) for s in get_spans(span, self.span_getter)]

        n_sents: int = self.context_sents
        n_words = self.context_words

        start = span.start - n_words
        end = span.end + n_words

        if n_sents > 0:
            if n_sents == 1:
                sent = span.sent
                min_start_sent = sent.start
                max_end_sent = sent.end
            else:
                sents = list(span.doc.sents) if n_sents > 1 else []
                sent_i = sents.index(span.sent)
                min_start_sent = sents[max(0, sent_i - n_sents)].start
                max_end_sent = sents[min(len(sents) - 1, sent_i + n_sents)].end
            start = max(0, min(start, min_start_sent))
            end = min(len(span.doc), max(end, max_end_sent))

        return span.doc[start:end]
