import copy
import re
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from confit import VisibleDeprecationWarning
from loguru import logger
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.pipes.base import BaseNERComponent, SpanSetterArg
from edsnlp.utils.collections import flatten_once
from edsnlp.utils.doc_to_text import get_text
from edsnlp.utils.span_getters import get_spans
from edsnlp.utils.typing import AsList  # noqa: F401

from . import models
from .models import FullConfig, SingleAssignModel, SingleConfig


@lru_cache(64)
def get_window(
    doclike: Union[Doc, Span], window: Tuple[int, int], limit_to_sentence: bool
):
    """
    Generate a window around the first parameter
    """
    start_limit = doclike.sent.start if limit_to_sentence else 0
    end_limit = doclike.sent.end if limit_to_sentence else len(doclike.doc)

    start = (
        max(doclike.start + window[0], start_limit)
        if window and window[0] is not None
        else start_limit
    )
    end = (
        min(doclike.end + window[1], end_limit)
        if window and window[0] is not None
        else end_limit
    )

    return doclike.doc[start:end]


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
    patterns : AsList[SingleConfig]
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
        if label is None and label_name is not None:
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

        # Configuration parsing
        self.patterns: Dict[str, SingleConfig] = copy.deepcopy(
            {pattern.source: pattern for pattern in patterns}
        )

        # Matchers for the anchors
        self.phrase_matcher = EDSPhraseMatcher(
            nlp.vocab,
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
        )
        self.regex_matcher = RegexMatcher(
            attr=attr,
            flags=regex_flags,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            alignment_mode=alignment_mode,
        )

        self.phrase_matcher.build_patterns(
            nlp=nlp,
            terms={
                source: {
                    "patterns": p.terms,
                }
                for source, p in self.patterns.items()
            },
        )
        self.regex_matcher.build_patterns(
            regex={
                source: {
                    "regex": p.regex,
                    "attr": p.regex_attr,
                    "flags": p.regex_flags,
                }
                for source, p in self.patterns.items()
            }
        )

        # Will contain the reduce mode (for each source and assign matcher)
        self.reduce_mode = {}

        # Will contain the name of the assign matcher from which
        # entity will be replaced (for each source)
        self.replace_key = {}

        for source, p in self.patterns.items():
            p: SingleConfig

            for exclude in p.exclude:
                exclude.matcher = RegexMatcher(
                    attr=exclude.regex_attr or p.regex_attr or self.attr,
                    flags=exclude.regex_flags or p.regex_flags or self.regex_flags,
                    ignore_excluded=ignore_excluded,
                    ignore_space_tokens=ignore_space_tokens,
                    alignment_mode="expand",
                )

                exclude.matcher.build_patterns(regex={"exclude": exclude.regex})

            for include in p.include:
                include.matcher = RegexMatcher(
                    attr=include.regex_attr or p.regex_attr or self.attr,
                    flags=include.regex_flags or p.regex_flags or self.regex_flags,
                    ignore_excluded=ignore_excluded,
                    ignore_space_tokens=ignore_space_tokens,
                    alignment_mode="expand",
                )

                include.matcher.build_patterns(regex={"include": include.regex})

            replace_key = None

            for assign in p.assign:
                assign.matcher = None
                if assign.regex:
                    assign.matcher = RegexMatcher(
                        attr=assign.regex_attr or p.regex_attr or self.attr,
                        flags=assign.regex_flags or p.regex_flags or self.regex_flags,
                        ignore_excluded=ignore_excluded,
                        ignore_space_tokens=ignore_space_tokens,
                        alignment_mode=alignment_mode,
                        span_from_group=True,
                    )

                    assign.matcher.build_patterns(
                        regex={assign.name: assign.regex},
                    )
                    assign.regex = assign.matcher

                if assign.replace_entity:
                    # We know that there is only one assign name
                    # with `replace_entity==True`
                    # from PyDantic validation
                    replace_key = assign.name

            self.reduce_mode[source] = {d.name: d.reduce_mode for d in p.assign}
            self.replace_key[source] = replace_key

    def set_extensions(self) -> None:
        """
        Define the extensions used by the component
        """
        super().set_extensions()
        if not Span.has_extension("assigned"):
            Span.set_extension("assigned", default=dict())
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)

    def filter_one(self, span: Span) -> Span:
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
        source = span.label_
        to_keep = True
        for exclude in self.patterns[source].exclude:
            snippet = exclude.window(span)

            if next(exclude.matcher(snippet, as_spans=True), None) is not None:
                to_keep = False
                logger.trace(f"Entity {span} was filtered out")
                break

        for include in self.patterns[source].include:
            snippet = include.window(span)

            if next(include.matcher(snippet, as_spans=True), None) is None:
                to_keep = False
                logger.trace(f"Entity {span} was filtered out")
                break

        if to_keep:
            return span

    def assign_one(self, span: Span) -> Span:
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
        Span
            Span with additional information
        """

        if span is None:
            return

        source = span.label_
        assigned_dict = models.AssignDict(reduce_mode=self.reduce_mode[source])
        replace_key = None

        all_assigned_list = []
        for assign in self.patterns[source].assign:
            assign: SingleAssignModel
            window = assign.window
            snippet = window(span)

            matcher: RegexMatcher = assign.matcher
            if matcher is not None:
                # Getting the matches
                assigned_list = list(matcher.match(snippet))
                assigned_list = [
                    (matched_span, matched_span, matcher.regex[0][0], assign)
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
                        matcher.regex[0][0],
                        assign,
                    )
                    for (matched_span, re_match) in assigned_list
                ]
            else:
                assigned_list = [
                    (matched_span, matched_span, assign.name, assign)
                    for matched_span in get_spans(snippet.doc, assign.span_getter)
                    if matched_span.start >= snippet.start
                    and matched_span.end <= snippet.end
                ]

            # assigned_list now contains tuples with
            # - the first element being the span extracted from the group
            # - the second element being the full match

            if assign.required and not assigned_list:
                logger.trace(f"Entity {span} was filtered out")
                return

            all_assigned_list.extend(assigned_list)

        for assigned in all_assigned_list:
            if assigned is None:
                continue
            group_span, full_match_span, value_key, assign = assigned
            if assign.replace_entity:
                replace_key = value_key

            # Using he overridden `__setitem__` method from AssignDict here:
            assigned_dict[value_key] = {
                "span": full_match_span,  # Full span
                "value_span": group_span,  # Span of the group
                "value_text": get_text(
                    group_span,
                    attr=self.patterns[source].regex_attr or self.attr,
                    ignore_excluded=self.ignore_excluded,
                ),  # Text of the group
            }
            logger.trace(f"Assign key {value_key} matched on entity {span}")

        if replace_key is None and self.replace_key[source] is not None:
            # There should have been a replacement, but none was found
            # So we discard the entity
            return

        # Entity replacement
        if replace_key is not None:
            replacables = assigned_dict[replace_key]["span"]
            kept_ents = (
                replacables if isinstance(replacables, list) else [replacables]
            ).copy()

            if self.include_assigned:
                # We look for the closest
                closest = min(
                    kept_ents,
                    key=lambda e: abs(e.start - span.start),
                )
                kept_ents.remove(closest)

                expandables = list(
                    flatten_once(
                        [
                            a["span"]
                            for k, a in assigned_dict.items()
                            if k != replace_key
                        ]
                    )
                ) + [span, closest]

                closest = Span(
                    span.doc,
                    min(span.start for span in expandables if span is not None),
                    max(span.end for span in expandables if span is not None),
                    span.label_,
                )

                kept_ents.append(closest)
                kept_ents.sort(key=lambda e: e.start)

            for replaced in kept_ents:
                # Propagating attributes from the anchor
                replaced._.source = source
                replaced.label_ = self.label

        else:
            # Entity expansion
            expandables = [
                s
                for s in flatten_once([a["span"] for a in assigned_dict.values()])
                if s is not None
            ]

            if self.include_assigned and expandables:
                span = Span(
                    span.doc,
                    min(s.start for s in expandables + [span] if s is not None),
                    max(s.end for s in expandables + [span] if s is not None),
                    span.label_,
                )

            span._.source = source
            span.label_ = self.label
            kept_ents = [span]

        key = "value_span" if self.assign_as_span else "value_text"

        for idx, e in enumerate(kept_ents):
            e._.assigned = {
                k: v[key][idx]
                if ((k == replace_key) and self.reduce_mode[source][k] is None)
                else v[key]
                for k, v in assigned_dict.items()
            }

        yield from kept_ents

    def process_one(self, span):
        """
        Processes one span, applying both the filters and the assignments

        Parameters
        ----------
        span:
            spaCy Span object

        Yields
        ------
        span:
            Filtered spans, with optional assignments
        """
        filtered = self.filter_one(span)
        yield from self.assign_one(filtered)

    def process(self, doc: Doc) -> List[Span]:
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

        matches = self.phrase_matcher(doc, as_spans=True)
        regex_matches = list(self.regex_matcher(doc, as_spans=True))

        spans = (*matches, *regex_matches)
        for span in spans:
            yield from self.process_one(span)

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
