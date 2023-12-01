from collections import defaultdict
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.utils.filter import filter_spans


class BaseComponent:
    """
    The `BaseComponent` adds a `set_extensions` method,
    called at the creation of the object.

    It helps decouple the initialisation of the pipeline from
    the creation of extensions, and is particularly usefull when
    distributing EDSNLP on a cluster, since the serialisation mechanism
    imposes that the extensions be reset.
    """

    def __init__(self, nlp: Language = None, name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = nlp
        self.name = name
        self.set_extensions()

    def set_extensions(self):
        """
        Set `Doc`, `Span` and `Token` extensions.
        """
        Span.set_extension(
            "value",
            getter=lambda span: span._.get(span.label_)
            if span._.has(span.label_)
            else None,
            force=True,
        )

    def get_spans(self, doc: Doc):
        """
        Returns sorted spans of interest according to the
        possible value of `on_ents_only`.
        Includes `doc.ents` by default, and adds eventual SpanGroups.
        """
        ents = list(doc.ents) + list(doc.spans.get("discarded", []))

        on_ents_only = getattr(self, "on_ents_only", None)

        if isinstance(on_ents_only, str):
            on_ents_only = [on_ents_only]
        if isinstance(on_ents_only, (set, list)):
            for spankey in set(on_ents_only) & set(doc.spans.keys()):
                ents.extend(doc.spans.get(spankey, []))

        return sorted(list(set(ents)), key=(attrgetter("start", "end")))

    def _boundaries(
        self, doc: Doc, terminations: Optional[List[Span]] = None
    ) -> List[Tuple[int, int]]:
        """
        Create sub sentences based sentences and terminations found in text.

        Parameters
        ----------
        doc:
            spaCy Doc object
        terminations:
            List of tuples with (match_id, start, end)

        Returns
        -------
        boundaries:
            List of tuples with (start, end) of spans
        """

        if terminations is None:
            terminations = []

        sent_starts = [sent.start for sent in doc.sents]
        termination_starts = [t.start for t in terminations]

        starts = sent_starts + termination_starts + [len(doc)]

        # Remove duplicates
        starts = list(set(starts))

        # Sort starts
        starts.sort()

        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]

        return boundaries


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
    if callable(span_getter):
        yield from span_getter(doc)
        return
    for key, span_filter in span_getter.items():
        candidates = doc.spans.get(key, ()) if key != "ents" else doc.ents
        if span_filter is True:
            yield from candidates
        else:
            for span in candidates:
                if span.label_ in span_filter:
                    yield span


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
    def validate(cls, value: Union[SeqStr, Dict[str, SpanFilter]]) -> SpanSetter:
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
    def validate(cls, value: Union[SeqStr, Dict[str, SpanFilter]]) -> SpanSetter:
        return validate_span_setter(value)


class BaseNERComponent(BaseComponent):
    def __init__(
        self,
        nlp: Language = None,
        name: str = None,
        *args,
        span_setter: SpanSetterArg,
        **kwargs,
    ):
        super().__init__(nlp, name, *args, **kwargs)
        self.span_setter: SpanSetter = validate_span_setter(span_setter)  # type: ignore

    def set_spans(self, doc, matches):
        if callable(self.span_setter):
            self.span_setter(doc, matches)
        else:

            match_all = []
            label_to_group = defaultdict(list)
            for name, spans_filter in self.span_setter.items():
                if name != "ents":
                    doc.spans.setdefault(name, [])
                if spans_filter:
                    if spans_filter is True:
                        match_all.append(name)
                    else:
                        for label in spans_filter:
                            label_to_group[label].append(name)

            new_ents = [] if "ents" in self.span_setter else None

            for span in matches:
                for group in match_all + label_to_group[span.label_]:
                    if group == "ents":
                        new_ents.append(span)
                    else:
                        doc.spans[group].append(span)
            if new_ents is not None:
                doc.ents = filter_spans((*new_ents, *doc.ents))
        return doc


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
