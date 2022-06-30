import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

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
    ):
        self.name = name
        self.nlp = nlp
        self.attr = attr
        self.assign_at_span = assign_as_span
        self.ignore_excluded = ignore_excluded
        self.alignment_mode = alignment_mode
        self.regex_flags = regex_flags

        patterns = models.FullConfig.parse_obj(patterns).__root__

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

        self.exclude_matchers = []
        self.assign_matchers = []

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

                self.exclude_matchers.append(
                    dict(
                        matcher=exclude_matcher,
                        window=exclude["window"],
                    )
                )

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

                self.assign_matchers.append(
                    dict(
                        matcher=assign_matcher,
                        window=assign["window"],
                        expand_entity=assign["expand_entity"],
                    )
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

        spans = (*matches, *regex_matches)
        spans = self.filter(spans)
        spans = self.assign(spans)
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
            for matcher in self.exclude_matchers:

                window = matcher["window"]
                snippet = get_window(
                    doclike=ent,
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
                    break

            if to_keep:
                yield ent

    def assign(self, spans: List[Span]) -> List[Span]:
        """
        Get additional information in the context
        of each entity. This function will populate two custom attributes:

        - `ent._.source`
        - `ent._.assigned`, a dictionary with all retrieved informations

        Parameters
        ----------
        spans : List[Span]
            Spans to enrich

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
            assigned_dict = {}
            for matcher in self.assign_matchers:

                attr = (
                    self.patterns[source].regex_attr or matcher["matcher"].default_attr
                )
                window = matcher["window"]
                expand_entity = matcher["expand_entity"]

                snippet = get_window(
                    doclike=ent,
                    window=window,
                )
                assigned_list = list(matcher["matcher"](snippet, as_spans=True))

                if not assigned_list:

                    continue

                for assigned in assigned_list:
                    if assigned is None:
                        continue
                    assigned_dict[assigned.label_] = {
                        "span": assigned,
                        assigned.label_: get_text(
                            assigned,
                            attr=attr,
                            ignore_excluded=self.ignore_excluded,
                        ),
                        "expand_entity": expand_entity,
                    }

            expandables = [
                a["span"]
                for a in assigned_dict.values()
                if a.get("expand_entity", False)
            ]

            if expandables:

                ent = Span(
                    ent.doc,
                    ent.start,
                    ent.end,
                    ent.label_,
                )

            ent._.source = source
            ent.label_ = self.name

            if self.assign_at_span:
                ent._.assigned = {k: v["span"] for k, v in assigned_dict.items()}
            else:
                ent._.assigned = {k: v[k] for k, v in assigned_dict.items()}

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

        ents = list(self.process(doc))

        doc.spans[self.name] = ents

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
