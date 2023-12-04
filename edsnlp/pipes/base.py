import warnings
from operator import attrgetter
from typing import (
    List,
    Optional,
    Tuple,
)

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.utils.span_getters import (
    SpanGetter,  # noqa: F401
    SpanGetterArg,  # noqa: F401
    SpanSetter,
    SpanSetterArg,
    get_spans,  # noqa: F401
    set_spans,
    validate_span_getter,  # noqa: F401
    validate_span_setter,
)


def value_getter(span: Span):
    key = span._._get_key("value")
    if key in span.doc.user_data:
        return span.doc.user_data[key]
    return span._.get(span.label_) if span._.has(span.label_) else None


class BaseComponent:
    """
    The `BaseComponent` adds a `set_extensions` method,
    called at the creation of the object.

    It helps decouple the initialisation of the pipeline from
    the creation of extensions, and is particularly usefull when
    distributing EDSNLP on a cluster, since the serialisation mechanism
    imposes that the extensions be reset.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.set_extensions()

    def set_extensions(self):
        """
        Set `Doc`, `Span` and `Token` extensions.
        """
        if Span.has_extension("value"):
            if Span.get_extension("value")[2] is not value_getter:
                warnings.warn(
                    "A Span extension 'value' already exists with a different getter. "
                    "Keeping the existing extension, but some components of edsnlp may "
                    "not work as expected."
                )
            return
        Span.set_extension(
            "value",
            getter=value_getter,
        )

    def get_spans(self, doc: Doc):  # noqa: F811
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

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.set_extensions()


class BaseNERComponent(BaseComponent):
    span_setter: SpanSetter

    def __init__(
        self,
        nlp: PipelineProtocol = None,
        name: str = None,
        *args,
        span_setter: SpanSetterArg,
        **kwargs,
    ):
        super().__init__(nlp, name, *args, **kwargs)
        self.span_setter: SpanSetter = validate_span_setter(span_setter)  # type: ignore

    def set_spans(self, doc, matches):
        return set_spans(doc, matches, self.span_setter)
