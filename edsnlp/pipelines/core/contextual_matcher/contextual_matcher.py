import re
from collections import defaultdict
from functools import lru_cache
from operator import attrgetter
from typing import Any, Dict, List, Tuple, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.lists import flatten

from . import models


@lru_cache(64)
def get_window(
    doclike: Union[Doc, Span],
    window: Tuple[int, int],
):

    return doclike.doc[
        max(doclike.start + window[0], doclike.sent.start) : min(
            doclike.end + window[1], doclike.sent.end
        )
    ]


class ContextualMatcher(BaseComponent):
    """
    Allows additional matching in the surrounding context of the main match group,
    for qualification/filtering.

    Parameters
    ----------
    nlp : Language
        spaCy `Language` object.
    name : str
        The name of the pipe
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
        assign_as_span: bool,
        alignment_mode: str,
        attr: str,
        regex_flags: Union[re.RegexFlag, int],
        ignore_excluded: bool,
        include_assigned: bool,
    ):
        self.name = name
        self.nlp = nlp
        self.attr = attr
        self.assign_as_span = assign_as_span
        self.ignore_excluded = ignore_excluded
        self.alignment_mode = alignment_mode
        self.regex_flags = regex_flags
        self.include_assigned = include_assigned

        # Configuration parsing
        patterns = models.FullConfig.parse_obj(patterns).__root__
        self.patterns = {pattern.source: pattern for pattern in patterns}

        # Matchers for the anchors
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

        self.exclude_matchers = defaultdict(
            list
        )  # Will contain all the exclusion matchers
        self.assign_matchers = defaultdict(list)  # Will contain all the assign matchers

        # Will contain the reduce mode (for each source and assign matcher)
        self.reduce_mode = {}

        # Will contain the name of the assign matcher from which
        # entity will be replaced (for each source)
        self.replace_key = {}

        for source, p in self.patterns.items():

            p = p.dict()

            for exclude in p["exclude"]:

                exclude_matcher = RegexMatcher(
                    attr=p["regex_attr"] or self.attr,
                    flags=p["regex_flags"] or self.regex_flags,
                    ignore_excluded=ignore_excluded,
                    alignment_mode="expand",
                )

                exclude_matcher.build_patterns(regex={"exclude": exclude["regex"]})

                self.exclude_matchers[source].append(
                    dict(
                        matcher=exclude_matcher,
                        window=exclude["window"],
                    )
                )

            replace_key = None

            for assign in p["assign"]:

                assign_matcher = RegexMatcher(
                    attr=p["regex_attr"] or self.attr,
                    flags=p["regex_flags"] or self.regex_flags,
                    ignore_excluded=ignore_excluded,
                    alignment_mode=alignment_mode,
                    span_from_group=True,
                )

                assign_matcher.build_patterns(
                    regex={assign["name"]: assign["regex"]},
                )

                self.assign_matchers[source].append(
                    dict(
                        name=assign["name"],
                        matcher=assign_matcher,
                        window=assign["window"],
                        replace_entity=assign["replace_entity"],
                        reduce_mode=assign["reduce_mode"],
                    )
                )

                if assign["replace_entity"]:
                    # We know that there is only one assign name
                    # with `replace_entity==True`
                    # from PyDantic validation
                    replace_key = assign["name"]

            self.replace_key[source] = replace_key

            self.reduce_mode[source] = {
                d["name"]: d["reduce_mode"] for d in self.assign_matchers[source]
            }

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        if not Span.has_extension("assigned"):
            Span.set_extension("assigned", default=dict())
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)

    def filter_one(self, span: Span) -> Span:
        """
        Filter extracted entity based on the "exclusion filter" mentionned
        in the configuration

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
        for matcher in self.exclude_matchers[source]:

            window = matcher["window"]
            snippet = get_window(
                doclike=span,
                window=window,
            )

            if (
                next(
                    matcher["matcher"](snippet, as_spans=True),
                    None,
                )
                is not None
            ):
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
            yield from []
            return

        source = span.label_
        assigned_dict = models.AssignDict(reduce_mode=self.reduce_mode[source])
        replace_key = None

        for matcher in self.assign_matchers[source]:

            attr = self.patterns[source].regex_attr or matcher["matcher"].default_attr
            window = matcher["window"]
            replace_entity = matcher["replace_entity"]  # Boolean

            snippet = get_window(
                doclike=span,
                window=window,
            )

            # Getting the matches
            assigned_list = list(matcher["matcher"].match(snippet))

            assigned_list = [
                (span, span)
                if not match.groups()
                else (
                    span,
                    create_span(
                        doclike=snippet,
                        start_char=match.start(0),
                        end_char=match.end(0),
                        key=matcher["matcher"].regex[0][0],
                        attr=matcher["matcher"].regex[0][2],
                        alignment_mode=matcher["matcher"].regex[0][4],
                        ignore_excluded=matcher["matcher"].regex[0][3],
                    ),
                )
                for (span, match) in assigned_list
            ]

            # assigned_list now contains tuples with
            # - the first element being the span extracted from the group
            # - the second element being the full match

            if not assigned_list:  # No match was found
                continue

            for assigned in assigned_list:
                if assigned is None:
                    continue
                if replace_entity:
                    replace_key = assigned[1].label_

                # Using he overrid `__setitem__` method from AssignDict here:
                assigned_dict[assigned[1].label_] = {
                    "span": assigned[1],  # Full span
                    "value_span": assigned[0],  # Span of the group
                    "value_text": get_text(
                        assigned[0],
                        attr=attr,
                        ignore_excluded=self.ignore_excluded,
                    ),  # Text of the group
                }
                logger.trace(f"Assign key {matcher['name']} matched on entity {span}")
        if replace_key is None and self.replace_key[source] is not None:
            # There should have been a replacement, but none was found
            # So we discard the entity
            yield from []
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

                expandables = flatten(
                    [a["span"] for k, a in assigned_dict.items() if k != replace_key]
                ) + [span, closest]

                closest = Span(
                    span.doc,
                    min(expandables, key=attrgetter("start")).start,
                    max(expandables, key=attrgetter("end")).end,
                    span.label_,
                )

                kept_ents.append(closest)
                kept_ents.sort(key=attrgetter("start"))

            for replaced in kept_ents:
                # Propagating attributes from the anchor
                replaced._.source = source
                replaced.label_ = self.name

        else:

            # Entity expansion
            expandables = flatten([a["span"] for a in assigned_dict.values()])

            if self.include_assigned and expandables:

                span = Span(
                    span.doc,
                    min(expandables + [span], key=attrgetter("start")).start,
                    max(expandables + [span], key=attrgetter("end")).end,
                    span.label_,
                )

            span._.source = source
            span.label_ = self.name
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
        regex_matches = self.regex_matcher(doc, as_spans=True)

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

        ents = list(self.process(doc))

        doc.spans[self.name] = ents

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
