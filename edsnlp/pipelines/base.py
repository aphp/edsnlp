from collections import defaultdict
from operator import attrgetter
from typing import (
    List,
    Optional,
    Tuple,
)

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.span_getters import (
    SpanGetter,  # noqa: F401
    SpanGetterArg,  # noqa: F401
    SpanSetter,
    SpanSetterArg,
    get_spans,  # noqa: F401
    validate_span_getter,  # noqa: F401
    validate_span_setter,
)


class BaseComponent:
    """
    The `BaseComponent` adds a `set_extensions` method,
    called at the creation of the object.

    It helps decouple the initialisation of the pipeline from
    the creation of extensions, and is particularly usefull when
    distributing EDSNLP on a cluster, since the serialisation mechanism
    imposes that the extensions be reset.
    """

    def __init__(self, nlp: PipelineProtocol = None, name: str = None, *args, **kwargs):
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
