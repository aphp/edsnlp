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
        alignment_mode: str = "expand",
        attr: str = "TEXT",
        ignore_excluded: bool = False,
    ):

        super().__init__(
            alignment_mode=alignment_mode, attr=attr, ignore_excluded=ignore_excluded
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
    nlp : Language
        spaCy `Language` object.
    regex_config : Dict[str, Any]
        Configuration for the main expression.
    window : int
        Number of tokens to consider before and after the main expression.
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    verbose : int
        Verbosity level, useful for debugging.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        name: str,
        nlp: Language,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
        alignment_mode: str,
        attr: str,
        ignore_excluded: bool,
    ):

        self.name = name
        self.nlp = nlp
        self.attr = attr
        self.ignore_excluded = ignore_excluded
        self.alignment_mode = alignment_mode

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

                exclude_matcher = RegexMatcher(
                    attr=p["regex_attr"] or self.attr,
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

                assign_matcher = GroupRegexMatcher(
                    attr=p["regex_attr"] or self.attr,
                    ignore_excluded=ignore_excluded,
                    alignment_mode=alignment_mode,
                )

                assign_matcher.build_patterns(regex=p["assign"][side]["regex"])
                self.assign_matchers[side][source] = dict(
                    matcher=assign_matcher,
                    window=p["assign"][side]["window"],
                    expand_entity=p["assign"][side]["expand_entity"],
                )

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        if not Doc.has_extension("my_ents"):
            Doc.set_extension("my_ents", default=[])
        if not Span.has_extension("assigned"):
            Span.set_extension("assigned", default=[])
        if not Span.has_extension("source"):
            Span.set_extension("source", default=None)

        if not Span.has_extension("before_extract"):
            Span.set_extension("before_extract", default=None)
        if not Span.has_extension("after_extract"):
            Span.set_extension("after_extract", default=None)

        if not Span.has_extension("window"):
            Span.set_extension("window", default=None)

        if not Span.has_extension("before_snippet"):
            Span.set_extension("before_snippet", default=None)
        if not Span.has_extension("after_snippet"):
            Span.set_extension("after_snippet", default=None)

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
        print("SPANS", spans)
        spans = list(self.filter(spans))
        print("SPANS", spans)
        spans = list(self.assign(spans))
        return spans

    def filter(self, spans: List[Span]) -> List[Span]:
        for ent in spans:
            to_keep = True
            for side in self.sides:
                source = ent.label_
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
        for ent in spans:
            source = ent.label_
            for side in self.sides:
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

                ent._.assigned.extend(
                    [
                        {
                            "span": assigned,
                            "value": get_text(
                                assigned,
                                attr=attr,
                                ignore_excluded=self.ignore_excluded,
                            ),
                            "title": assigned.label_,
                            "source": ent.label_,
                            "side": side,
                        }
                        for assigned in assigned_list
                        if assigned is not None
                    ],
                )
                print(ent._.assigned)
                if expand_entity:
                    assigned = ent._.assigned
                    min_start = min(
                        [a["span"].start_char for a in assigned] + [ent.start_char]
                    )
                    max_end = max(
                        [a["span"].end_char for a in assigned] + [ent.end_char]
                    )
                    ent = create_span(
                        doclike=ent.doc,
                        start_char=min_start,
                        end_char=max_end,
                        key=ent.label_,
                        attr=attr,
                        alignment_mode=self.alignment_mode,
                        ignore_excluded=self.ignore_excluded,
                    )
                    ent._.assigned = assigned

            ent._.source = source
            ent.label_ = self.name

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

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc

    def _postprocessing_pipeline(self, ents: List[Span]):
        # add a window within the sentence around entities
        ents = [self._add_window(ent) for ent in ents]

        # Remove entities based on the snippet located just before and after the entity
        ents = filter(self._exclude_filter, ents)

        # Extract informations from the entity's context via regex
        ents = [self._snippet_extraction(ent) for ent in ents]

        return ents

    def _add_window(self, ent: Span) -> Span:
        ent._.window = ent.doc[
            max(ent.start - self.window, ent.sent.start) : min(
                ent.end + self.window, ent.sent.end
            )
        ]

        # include the entity in the snippets so that we can extract
        # the number when it is attached to the word, e.g. "3PA"
        ent._.before_snippet = ent.doc[
            max(ent.start - self.window, ent.sent.start) : ent.end
        ]
        ent._.after_snippet = ent.doc[
            ent.start : min(ent.end + self.window, ent.sent.end)
        ]
        return ent

    def get_text(self, span: Span, label) -> str:
        attr = self.regex_config[label].get("attr", self.attr)

        return get_text(
            doclike=span,
            attr=attr,
            ignore_excluded=self.ignore_excluded,
        )

    def _exclude_filter(self, ent: Span) -> Span:
        label = ent.label_

        before_exclude = self.regex_config[label].get("before_exclude", None)
        after_exclude = self.regex_config[label].get("after_exclude", None)

        if before_exclude is not None:
            t = ent._.before_snippet
            t = self.get_text(t, label)
            if re.compile(before_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        f"excluded (before) string: {t} - pattern {before_exclude}"
                    )
                return False

        if after_exclude is not None:
            t = ent._.after_snippet
            t = self.get_text(t, label)
            if re.compile(after_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        f"excluded (after) string: {t} - pattern {after_exclude}"
                    )
                return False

        return True

    def _snippet_extraction(self, ent: Span) -> Span:
        label = ent.label_

        before_extract = self.regex_config[label].get("before_extract", [])
        after_extract = self.regex_config[label].get("after_extract", [])

        if type(before_extract) == str:
            before_extract = [before_extract]
        if type(after_extract) == str:
            after_extract = [after_extract]

        t = ent._.before_snippet
        t = self.get_text(t, label)
        ent._.before_extract = []
        for pattern in before_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.before_extract.append(match.groups()[0] if match else None)

        t = ent._.after_snippet
        t = self.get_text(t, label)
        ent._.after_extract = []
        for pattern in after_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.after_extract.append(match.groups()[0] if match else None)

        return ent


def _check_regex_config(regex_config):
    for k, v in regex_config.items():
        if type(v) is not dict:
            raise TypeError(
                f"The value of the key {k} is of type {type(v)}, but a dict is expected"
            )

        single_group_regex_keys = ["before_extract", "after_extract"]

        for single_group_regex_key in single_group_regex_keys:
            if single_group_regex_key in v:
                # ensure it is a list
                if type(v[single_group_regex_key]) is not list:
                    v[single_group_regex_key] = [v[single_group_regex_key]]

                for i, regex in enumerate(v[single_group_regex_key]):
                    n_groups = re.compile(regex).groups

                    if n_groups == 0:
                        # Adding grouping parenthesis
                        v[single_group_regex_key][i] = r"(" + regex + r")"
                    elif n_groups != 1:
                        # Accepting only 1 group per regex
                        raise ValueError(
                            f"The RegEx for {repr(k)} ({repr(regex)}) "
                            f"stored in {repr(single_group_regex_key)} "
                            f"contains {n_groups} capturing groups, 1 expected"
                        )

    return regex_config
