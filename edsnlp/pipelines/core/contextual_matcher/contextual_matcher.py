import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import models


@lru_cache(64)
def get_window(
    doclike: Union[Doc, Span],
    window: int,
    side: str,
):
    if side == "before":
        snippet = doclike.doc[
            max(doclike.start - window, doclike.sent.start) : doclike.end
        ]
    elif side == "after":
        snippet = doclike.doc[
            doclike.start : min(doclike.end + window, doclike.sent.end)
        ]
    return snippet


class GroupRegexMatcher(RegexMatcher):
    def __init__(
        self,
        alignment_mode: str,
        attr: str,
        flags: models.Flags,
        ignore_excluded: bool,
    ):

        super().__init__(
            alignment_mode=alignment_mode,
            attr=attr,
            ignore_excluded=ignore_excluded,
            flags=flags,
        )

    def match(
        self,
        doclike: Union[Doc, Span],
    ) -> Tuple[Span, re.Match]:
        """
        Iterates on the matches.

        Parameters
        ----------
        doclike:
            spaCy Doc or Span object to match on.

        Yields
        -------
        span:
            A match.
        """

        for key, patterns, attr, ignore_excluded, alignment_mode in self.regex:
            text = get_text(doclike, attr, ignore_excluded)

            for pattern in patterns:
                match = pattern.search(text)

                if match is None:
                    continue

                logger.trace(f"Matched a regex from {key}: {repr(match.group())}")
                span_char = match.span(1)

                if span_char[0] < 0:  # Group didn't match
                    continue

                span = create_span(
                    doclike=doclike,
                    start_char=span_char[0],
                    end_char=span_char[1],
                    key=key,
                    attr=attr,
                    alignment_mode=alignment_mode,
                    ignore_excluded=ignore_excluded,
                )

                yield span


class ContextualMatcher(BaseComponent):
    """
    Allows additional matching in the surrounding context of the main match group,
    for qualification/filtering.

    Parameters
    ----------
    name : str
        The name of the pipe
    nlp : Language
        spaCy `Language` object.
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]]
        The configuration dictionary
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    alignment_mode : str
        Overwrite alignment mode.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
        alignment_mode: str,
        attr: str,
        regex_flags: Union[re.RegexFlag, int],
        ignore_excluded: bool,
    ):
        self.name = name
        self.nlp = nlp
        self.attr = attr
        self.ignore_excluded = ignore_excluded
        self.alignment_mode = alignment_mode
        self.regex_flags = regex_flags

        if isinstance(patterns, dict):
            patterns = [patterns]

        patterns = [models.SingleConfig.parse_obj(pattern) for pattern in patterns]

        assert len([pattern.source for pattern in patterns]) == len(
            set([pattern.source for pattern in patterns])
        ), "Each `source` field must be different !"

        self.patterns = {pattern.source: pattern for pattern in patterns}

        self.phrase_matcher = EDSPhraseMatcher(
            self.nlp.vocab,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )
        self.regex_matcher = RegexMatcher(
            attr=attr,
            flags=regex_flags,
            ignore_excluded=ignore_excluded,
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

        self.sides = ["before", "after"]

        self.exclude_matchers = {side: dict() for side in self.sides}
        self.assign_matchers = {side: dict() for side in self.sides}

        for side in self.sides:
            for source, p in self.patterns.items():

                p = p.dict()

                exclude_matcher = None

                if p["exclude"][side]["regex"]:
                    exclude_matcher = RegexMatcher(
                        attr=p["regex_attr"] or self.attr,
                        flags=p["regex_flags"] or self.regex_flags,
                        ignore_excluded=ignore_excluded,
                        alignment_mode="expand",
                    )
                    exclude_matcher.build_patterns(
                        regex=dict(
                            exclude=p["exclude"][side]["regex"],
                        )
                    )

                self.exclude_matchers[side][source] = dict(
                    matcher=exclude_matcher,
                    window=p["exclude"][side]["window"],
                )

                assign_matcher = None

                if p["assign"][side]["regex"]:

                    assign_matcher = GroupRegexMatcher(
                        attr=p["regex_attr"] or self.attr,
                        flags=p["regex_flags"] or self.regex_flags,
                        ignore_excluded=ignore_excluded,
                        alignment_mode=alignment_mode,
                    )

                    assign_matcher.build_patterns(
                        regex=p["assign"][side]["regex"],
                    )

                self.assign_matchers[side][source] = dict(
                    matcher=assign_matcher,
                    window=p["assign"][side]["window"],
                    expand_entity=p["assign"][side]["expand_entity"],
                )

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        if not Span.has_extension("assigned"):
            Span.set_extension("assigned", default=dict())
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)

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
        regex_matches = self.regex_matcher(doc, as_spans=True)

        spans = list(matches) + list(regex_matches)
        spans = list(self.filter(spans))
        spans = list(self.assign(spans))
        return spans

    def filter(self, spans: List[Span]) -> List[Span]:
        """
        Filter extracted entities based on the "exclusion filter" mentionned
        in the configuration

        Parameters
        ----------
        spans : List[Span]
            Spans to filter

        Yields
        ------
        Iterator[List[Span]]
            All spans that passed the filtering step
        """

        for ent in spans:
            to_keep = True
            for side in self.sides:
                source = ent.label_

                if (
                    self.exclude_matchers[side][source]["matcher"] is None
                ):  # Nothing to match
                    continue

                window = self.exclude_matchers[side][source]["window"]
                snippet = get_window(
                    doclike=ent,
                    window=window,
                    side=side,
                )

                if (
                    next(
                        self.exclude_matchers[side][source]["matcher"].match(snippet),
                        None,
                    )
                    is not None
                ):
                    to_keep = False
                    break

            if to_keep:
                yield ent

    def assign(self, spans: List[Span]) -> List[Span]:
        """
        Get additional information in the context
        of each entity. This funciton will populate two custom attributes:

        - `ent._.source`
        - `ent._.assigned`, a dictionary with all retrieved informations

        Parameters
        ----------
        spans : List[Span]
            _description_

        Parameters
        ----------
        spans : List[Span]
            Spans to filter

        Yields
        ------
        Iterator[List[Span]]
            All spans with additional informations
        """

        for ent in spans:
            source = ent.label_
            for side in self.sides:
                if (
                    self.assign_matchers[side][source]["matcher"] is None
                ):  # Nothing to match
                    continue
                attr = (
                    self.patterns[source].regex_attr
                    or self.assign_matchers[side][source]["matcher"].default_attr
                )
                window = self.assign_matchers[side][source]["window"]
                expand_entity = self.assign_matchers[side][source]["expand_entity"]

                snippet = get_window(
                    doclike=ent,
                    window=window,
                    side=side,
                )
                assigned_list = list(
                    self.assign_matchers[side][source]["matcher"].match(snippet)
                )

                if not assigned_list:

                    continue

                for assigned in assigned_list:
                    if assigned is None:
                        continue
                    ent._.assigned[assigned.label_] = {
                        "span": assigned,
                        assigned.label_: get_text(
                            assigned,
                            attr=attr,
                            ignore_excluded=self.ignore_excluded,
                        ),
                        "expand_entity": expand_entity,
                    }

            assigned = ent._.assigned
            expandables = [
                a["span"] for a in assigned.values() if a.get("expand_entity", False)
            ]

            if expandables:

                min_start = min([a.start_char for a in expandables] + [ent.start_char])
                max_end = max([a.end_char for a in expandables] + [ent.end_char])
                ent = create_span(
                    doclike=ent.doc,
                    start_char=min_start,
                    end_char=max_end,
                    key=ent.label_,
                    attr=attr,
                    alignment_mode=self.alignment_mode,
                    ignore_excluded=self.ignore_excluded,
                )

            ent._.source = source
            ent.label_ = self.name
            ent._.assigned = {k: v[k] for k, v in assigned.items()}

            yield ent

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

        ents = self.process(doc)

        doc.spans[self.name] = ents

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
