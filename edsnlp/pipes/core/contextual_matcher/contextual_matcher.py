import re
import warnings
from typing import Generator, Iterable, Optional, Union

from confit import VisibleDeprecationWarning, validate_arguments
from loguru import logger
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.pipes.base import BaseNERComponent, SpanSetterArg
from edsnlp.utils.doc_to_text import get_text
from edsnlp.utils.span_getters import get_spans

from .models import FullConfig, SingleAssignModel, SingleConfig


@validate_arguments()
class ContextualMatcher(BaseNERComponent):
    """
    Allows additional matching in the surrounding context of the main match group,
    for qualification/filtering.

    Parameters
    ----------
    nlp : PipelineProtocol
        spaCy `Language` object.
    name : Optional[str]
        The name of the pipe
    patterns : FullConfig
        ??? subdoc "The patterns to match"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleConfig
                options:
                    only_parameters: "no-header"
                    show_toc: false
    assign_as_span : bool
        Whether to store eventual extractions defined via the `assign` key as Spans
        or as string
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    ignore_space_tokens : bool
        Whether to skip space tokens during matching.
    alignment_mode : str
        Overwrite alignment mode.
    regex_flags : Union[re.RegexFlag, int]
        RegExp flags to use when matching, filtering and assigning (See
        [here](https://docs.python.org/3/library/re.html#flags))
    include_assigned : bool
        Whether to include (eventual) assign matches to the final entity
    label_name : Optional[str]
        Deprecated, use `label` instead. The label to assign to the matched entities
    label : str
        The label to assign to the matched entities
    span_setter : SpanSetterArg
        How to set matches on the doc
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: Optional[str] = "contextual_matcher",
        *,
        patterns: FullConfig,
        assign_as_span: bool = False,
        alignment_mode: str = "expand",
        attr: str = "NORM",
        regex_flags: Union[re.RegexFlag, int] = 0,
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        include_assigned: bool = False,
        label_name: Optional[str] = None,
        label: Optional[str] = None,
        span_setter: SpanSetterArg = {"ents": True},
    ):
        if label is None and label_name is not None:  # pragma: no cover
            warnings.warn(
                "`label_name` is deprecated, use `label` instead.",
                VisibleDeprecationWarning,
            )
            label = label_name
        if label is None:
            raise ValueError("`label` parameter is required.")
        self.label = label

        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        self.attr = attr
        self.assign_as_span = assign_as_span
        self.ignore_excluded = ignore_excluded
        self.ignore_space_tokens = ignore_space_tokens
        self.alignment_mode = alignment_mode
        self.regex_flags: Union[re.RegexFlag, int] = regex_flags
        self.include_assigned = include_assigned

        for pattern in patterns:
            phrase_matcher = EDSPhraseMatcher(
                nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                ignore_space_tokens=ignore_space_tokens,
            )
            phrase_matcher.build_patterns(
                nlp=nlp,
                terms={
                    "terms": {
                        "patterns": pattern.terms,
                    }
                },
            )
            pattern.phrase_matcher = phrase_matcher

            regex_matcher = RegexMatcher(
                attr=attr,
                flags=regex_flags,
                ignore_excluded=ignore_excluded,
                ignore_space_tokens=ignore_space_tokens,
                alignment_mode=alignment_mode,
            )
            regex_matcher.build_patterns(
                regex={
                    "regex": {
                        "regex": pattern.regex,
                        "attr": pattern.regex_attr,
                        "flags": pattern.regex_flags,
                    }
                }
            )
            pattern.regex_matcher = regex_matcher

            for exclude in pattern.exclude:
                if exclude.regex is not None:
                    matcher = RegexMatcher(
                        attr=exclude.regex_attr or pattern.regex_attr or self.attr,
                        flags=exclude.regex_flags
                        or pattern.regex_flags
                        or self.regex_flags,
                        ignore_excluded=ignore_excluded,
                        ignore_space_tokens=ignore_space_tokens,
                        alignment_mode="expand",
                    )
                    matcher.build_patterns(regex={"exclude": exclude.regex})
                    exclude.regex_matcher = matcher

            for include in pattern.include:
                if include.regex is not None:
                    matcher = RegexMatcher(
                        attr=include.regex_attr or pattern.regex_attr or self.attr,
                        flags=include.regex_flags
                        or pattern.regex_flags
                        or self.regex_flags,
                        ignore_excluded=ignore_excluded,
                        ignore_space_tokens=ignore_space_tokens,
                        alignment_mode="expand",
                    )
                    matcher.build_patterns(regex={"include": include.regex})
                    include.regex_matcher = matcher

            # replace_key = None

            for assign in pattern.assign:
                assign.regex_matcher = RegexMatcher(
                    attr=assign.regex_attr or pattern.regex_attr or self.attr,
                    flags=assign.regex_flags or pattern.regex_flags or self.regex_flags,
                    ignore_excluded=ignore_excluded,
                    ignore_space_tokens=ignore_space_tokens,
                    alignment_mode=alignment_mode,
                    span_from_group=True,
                )
                assign.regex_matcher.build_patterns(
                    regex={assign.name: assign.regex},
                )

        self.patterns = patterns

    def set_extensions(self) -> None:
        """
        Define the extensions used by the component
        """
        super().set_extensions()
        if not Span.has_extension("assigned"):
            Span.set_extension("assigned", default=dict())
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)

    def filter_one(self, span: Span, pattern) -> Optional[Span]:
        """
        Filter extracted entity based on the exclusion and inclusion filters of
        the configuration.

        Parameters
        ----------
        span : Span
            Span to filter

        Returns
        -------
        Optional[Span]
            None if the span was filtered, the span else
        """
        to_keep = True
        for exclude in pattern.exclude:
            snippet = exclude.window(span)

            if (
                exclude.regex_matcher is not None
                and any(
                    # check that it isn't inside in the anchor span
                    not (s.start >= span.start and s.end <= span.end)
                    for s in exclude.regex_matcher(snippet, as_spans=True)
                )
                or exclude.span_getter is not None
                and any(
                    # check that it isn't inside in the anchor span
                    not (s.start >= span.start and s.end <= span.end)
                    for s in get_spans(snippet, exclude.span_getter)
                )
            ):
                to_keep = False
                break

        for include in pattern.include:
            snippet = include.window(span)

            if (
                include.regex_matcher is not None
                and not any(
                    # check that it isn't inside in the anchor span
                    not (s.start >= span.start and s.end <= span.end)
                    for s in include.regex_matcher(snippet, as_spans=True)
                )
                or include.span_getter is not None
                and not any(
                    # check that it isn't inside in the anchor span
                    not (s.start >= span.start and s.end <= span.end)
                    for s in get_spans(snippet, include.span_getter)
                )
            ):
                to_keep = False
                break

        if to_keep:
            return span

    def assign_one(self, span: Span, pattern) -> Iterable[Span]:
        """
        Get additional information in the context
        of each entity. This function will populate two custom attributes:

        - `ent._.source`
        - `ent._.assigned`, a dictionary with all retrieved information

        Parameters
        ----------
        span : Span
            Span to enrich

        Returns
        -------
        List[Span]
            Spans with additional information
        """
        replace_key = None

        # Assigned matches is a list of tuples, each containing:
        # - the span matched by the "assign" regex (or returned by the span getter)
        # - the span corresponding to the match group of the regex (or the full match,
        #   ie same as above)
        assigned_dict = {}
        reduce_modes = {}
        attrs = {}

        for assign in pattern.assign:
            assign: SingleAssignModel
            window = assign.window
            snippet = window(span)
            reduce_modes[assign.name] = assign.reduce_mode
            matcher: RegexMatcher = assign.regex_matcher
            attrs[assign.name] = matcher.regex[0][2]
            if matcher is not None:
                # Getting the matches
                matches = list(matcher.match(snippet))
                assigned = [
                    (matched_span, matched_span)
                    if not re_match.groups()
                    else (
                        matched_span,
                        create_span(
                            doclike=snippet,
                            start_char=re_match.start(0),
                            end_char=re_match.end(0),
                            key=matcher.regex[0][0],
                            attr=matcher.regex[0][2],
                            alignment_mode=matcher.regex[0][5],
                            ignore_excluded=matcher.regex[0][3],
                            ignore_space_tokens=matcher.regex[0][4],
                        ),
                        # matcher.regex[0][0],
                    )
                    for (matched_span, re_match) in matches
                ]
            if assign.span_getter is not None:
                assigned = [
                    (matched_span, matched_span)
                    for matched_span in get_spans(snippet, assign.span_getter)
                    # if matched_span.start >= snippet.start
                    # and matched_span.end <= snippet.end
                ]

            if assign.required and not assigned:
                logger.trace(f"Entity {span} was filtered out")
                return []

            if len(assigned) == 0:
                continue

            if assign.replace_entity:
                replace_key = assign.name
            if assign.reduce_mode == "keep_first":  # closest
                assigned = [min(assigned, key=lambda e: abs(e[0].start - span.start))]
            elif assign.reduce_mode == "keep_last":
                assigned = [max(assigned, key=lambda e: abs(e[0].start - span.start))]

            assigned_dict[assign.name] = assigned

        # Several cases:
        # 1. should_have_replacement and include_assigned is True
        #    -> pick closest assigned span where replace = True
        #    ->
        if replace_key is not None:
            replacements = sorted(
                assigned_dict[replace_key],
                key=lambda e: abs(e[0].start - span.start),
            )
            assigned_dict[replace_key] = replacements

        ext = {
            n: None
            if reduce_modes[n] is not None and len(g) == 0
            else [s[0] for s in g][slice(None) if reduce_modes[n] is None else 0]
            if self.assign_as_span
            else [
                get_text(
                    s[0],
                    attr=attrs[n],
                    ignore_excluded=self.ignore_excluded,
                    ignore_space_tokens=self.ignore_space_tokens,
                )
                for s in g
            ][slice(None) if reduce_modes[n] is None else 0]
            for n, g in assigned_dict.items()
        }

        if replace_key is None:
            if self.include_assigned:
                merged = [span, *(x[1] for name, g in assigned_dict.items() for x in g)]
                span = Span(
                    span.doc,
                    min(s.start for s in merged),
                    max(s.end for s in merged),
                    span.label_,
                )
            span._.source = pattern.source
            span.label_ = self.label
            span._.assigned = ext
            new_spans = [span]
        else:
            if self.include_assigned:
                # we will merge spans from other assign groups + the main span
                # to the closest "most central" assign span.
                [closest_replacement, *rest_replacements] = assigned_dict[replace_key]
                other_spans = [
                    x[1]
                    for name, g in assigned_dict.items()
                    if name != replace_key
                    for x in g
                ]
                merged = [closest_replacement[1], span, *other_spans]
                span = Span(
                    span.doc,
                    min(s.start for s in merged),
                    max(s.end for s in merged),
                    span.label_,
                )
                new_spans = [span, *(s[1] for s in rest_replacements)]
            else:
                new_spans = [x[1] for x in assigned_dict[replace_key]]
            for idx, span in enumerate(new_spans):
                span._.source = pattern.source
                span.label_ = self.label
                span._.assigned = {
                    k: v[idx] if ((k == replace_key) and reduce_modes[k] is None) else v
                    for k, v in ext.items()
                }

        return new_spans

    def process_one(self, span: Span, pattern: SingleConfig):
        """
        Processes one span, applying both the filters and the assignments

        Parameters
        ----------
        span: Span
            Span object
        pattern: SingleConfig

        Yields
        ------
        span:
            Filtered spans, with optional assignments
        """
        span = self.filter_one(span, pattern)
        if span is not None:
            yield from self.assign_one(span, pattern)

    def process(self, doc: Doc) -> Generator[Span, None, None]:
        """
        Process the document, looking for named entities.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        List[Span]
            List of detected spans.
        """

        for pattern in self.patterns:
            for span in (
                *pattern.phrase_matcher(doc, as_spans=True),
                *pattern.regex_matcher(doc, as_spans=True),
                *(
                    get_spans(doc, pattern.span_getter)
                    if pattern.span_getter is not None
                    else []
                ),
            ):
                spans = list(self.process_one(span, pattern))
                yield from spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """

        spans = list(self.process(doc))
        self.set_spans(doc, spans)
        return doc
