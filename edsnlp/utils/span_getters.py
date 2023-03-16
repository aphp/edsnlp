from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from spacy import registry
from spacy.tokens import Doc, Span

Spans = List[Span]
SpanGroups = Dict[str, Spans]


class make_span_getter:
    def __init__(
        self,
        on_ents: Optional[Union[bool, Sequence[str]]] = None,
        on_spans_groups: Union[
            bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
        ] = False,
    ):

        """
        Make a span qualifier candidate getter function.

        Parameters
        ----------
        on_ents: Union[bool, Sequence[str]]
            Whether to look into `doc.ents` for spans to classify. If a list of strings
            is provided, only the span of the given labels will be considered. If None
            and `on_spans_groups` is False, labels mentioned in `label_constraints`
            will be used.
        on_spans_groups: Union[bool, Sequence[str], Mapping[str, Sequence[str]]]
            Whether to look into `doc.spans` for spans to classify:

            - If True, all span groups will be considered
            - If False, no span group will be considered
            - If a list of str is provided, only these span groups will be kept
            - If a mapping is provided, the keys are the span group names and the values
              are either a list of allowed labels in the group or True to keep them all
        """

        if not on_spans_groups and on_ents is None:
            on_ents = True

        self.on_ents = on_ents
        self.on_spans_groups = on_spans_groups

    def __call__(
        self,
        doc: Doc,
        return_origin: bool = False,
    ) -> Union[Tuple[Spans], Tuple[Spans, Optional[Spans], SpanGroups]]:
        flattened_spans = []
        span_groups = {}
        ents = None
        if self.on_ents:
            # /!\ doc.ents is not a list but a Span iterator, so to ensure referential
            # equality between the spans of `flattened_spans` and `ents`,
            # we need to convert it to a list to "extract" the spans first
            ents = list(doc.ents)
            if isinstance(self.on_ents, Sequence):
                flattened_spans.extend(
                    span for span in ents if span.label_ in self.on_ents
                )
            else:
                flattened_spans.extend(ents)

        if self.on_spans_groups:
            if isinstance(self.on_spans_groups, Mapping):
                for name, labels in self.on_spans_groups.items():
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
            elif isinstance(self.on_spans_groups, Sequence):
                for name in self.on_spans_groups:
                    span_groups[name] = list(doc.spans.get(name, ()))
                    flattened_spans.extend(span_groups[name])
            else:
                for name, spans_ in doc.spans.items():
                    # /!\ spans_ is not a list but a SpanGroup, so to ensure referential
                    # equality between the spans of `flattened_spans` and `span_groups`,
                    # we need to convert it to a list to "extract" the spans first
                    span_groups[name] = list(spans_)
                    flattened_spans.extend(span_groups[name])

        if return_origin:
            return flattened_spans, ents, span_groups
        else:
            return flattened_spans


registry.misc("eds.span_getter")(make_span_getter)
