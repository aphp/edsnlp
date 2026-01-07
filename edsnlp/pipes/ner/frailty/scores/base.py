import re
from typing import Any, Callable, Iterable, Optional, Tuple, Union

from loguru import logger
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher import ContextualMatcher
from edsnlp.pipes.core.contextual_matcher.models import FullConfig, SingleAssignModel
from edsnlp.utils.doc_to_text import get_text
from edsnlp.utils.filter import start_sort_key
from edsnlp.utils.span_getters import (
    IntersectionContextWindow,
    WordContextWindow,
    get_spans,
)


class FrailtyScoreMatcher(ContextualMatcher):
    """
    Matcher component to extract scores related to frailty evaluation."""

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str,
        domain: str,
        *,
        patterns: FullConfig,
        attr: str = "NORM",
        score_normalization: Union[str, Callable[[Span], Any]] = None,
        severity_assigner: Callable[
            [Union[str, Tuple[float, int], Tuple[int, int]]], Any
        ] = None,
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        flags: Union[re.RegexFlag, int] = 0,
        label: str = None,
        span_setter: Optional[SpanSetterArg] = None,
        include_assigned: bool = False,
    ):
        if label is None:
            raise ValueError("`label` parameter is required.")

        if span_setter is None:
            span_setter = {"ents": True, label: True}

        self.domain = domain

        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            assign_as_span=False,
            alignment_mode="expand",
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            attr=attr,
            regex_flags=flags,
            include_assigned=include_assigned,
            label=label,
            span_setter=span_setter,
        )

        if isinstance(score_normalization, str):
            self.score_normalization = registry.misc.get(score_normalization)
        else:
            self.score_normalization = score_normalization
        self.severity_assigner = severity_assigner

    def set_extensions(self):
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)
        if not Span.has_extension(self.domain):
            Span.set_extension(self.domain, default=None)

    def assign_one(self, span: Span, pattern) -> Iterable[Span]:
        """
        Overload # TODO doc

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

        # First we look for potential limits for our "real" assign context windows
        limit_window = None
        for limit_assign in pattern.assign:
            if "limit" not in limit_assign.name:
                continue
            window = limit_assign.window
            snippet = window(span)
            matcher: RegexMatcher = limit_assign.regex_matcher
            matches = list(matcher(snippet, as_spans=True))
            if len(matches) == 0:
                continue
            # We have found limiting elements, we must restrain the context windows
            matches = sorted(matches, key=start_sort_key)
            limit = matches[0].start - span.end
            if limit_window is None:
                limit_window = WordContextWindow(before=0, after=limit)
            elif limit < limit_window.after:
                limit_window.after = limit

        for assign in pattern.assign:
            if "limit" in assign.name:
                continue
            assign: SingleAssignModel
            window = assign.window
            if limit_window is not None:
                window = IntersectionContextWindow([window, limit_window])
            snippet = window(span)
            reduce_modes[assign.name] = assign.reduce_mode
            matcher: RegexMatcher = assign.regex_matcher
            attrs[assign.name] = matcher.regex[0][2]
            if matcher is not None:
                # Getting the matches
                matches = list(matcher.match(snippet))
                assigned = [
                    (
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
            n: (
                None
                if reduce_modes[n] is not None and len(g) == 0
                else (
                    [s[0] for s in g][slice(None) if reduce_modes[n] is None else 0]
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
                )
            )
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

    # TODO : add score normalization and post processing etc
    def process(self, doc: Doc) -> Iterable[Span]:
        for ent in super().process(doc):
            ent = self.score_normalization(ent)
            if ent is None:
                continue
            normalized_value = ent._.assigned.get("value", None)
            # value = ent._.assigned.get("value", None)
            # if value is None:
            #     continue
            # normalized_value = self.score_normalization(value)
            if normalized_value is not None:
                ent._.set(self.label, normalized_value)
                ent._.set(self.domain, self.severity_assigner(ent))
                yield ent


create_component = registry.factory.register(
    "eds.frailty_score",
)(FrailtyScoreMatcher)
