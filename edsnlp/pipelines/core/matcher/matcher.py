from enum import Enum
from typing import Any, Dict, List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.simstring import SimstringMatcher
from edsnlp.matchers.utils import Patterns
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans


class GenericTermMatcher(str, Enum):
    exact = "exact"
    simstring = "simstring"


class GenericMatcher(BaseComponent):
    """
    Provides a generic matcher component.

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    terms : Optional[Patterns]
        A dictionary of terms.
    regex : Optional[Patterns]
        A dictionary of regular expressions.
    attr : str
        The default attribute to use for matching.
        Can be overridden using the `terms` and `regex` configurations.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    term_matcher: GenericTermMatcher
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher class
    """

    def __init__(
        self,
        nlp: Language,
        terms: Optional[Patterns],
        regex: Optional[Patterns],
        attr: str,
        ignore_excluded: bool,
        term_matcher: GenericTermMatcher = GenericTermMatcher.exact,
        term_matcher_config: Dict[str, Any] = None,
    ):

        self.nlp = nlp

        self.attr = attr

        if term_matcher == GenericTermMatcher.exact:
            self.phrase_matcher = EDSPhraseMatcher(
                self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                **(term_matcher_config or {}),
            )
        elif term_matcher == GenericTermMatcher.simstring:
            self.phrase_matcher = SimstringMatcher(
                self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                **(term_matcher_config or {}),
            )
        else:
            raise ValueError(
                f"Algorithm {repr(term_matcher)} does not belong to"
                f" known matcher [exact, simstring]."
            )

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms)
        self.regex_matcher.build_patterns(regex=regex)

        self.set_extensions()

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object.

        Returns
        -------
        spans:
            List of Spans returned by the matchers.
        """

        matches = self.phrase_matcher(doc, as_spans=True)
        regex_matches = self.regex_matcher(doc, as_spans=True)

        spans = list(matches) + list(regex_matches)

        return spans

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
        matches = self.process(doc)

        for span in matches:
            if span.label_ not in doc.spans:
                doc.spans[span.label_] = []
            doc.spans[span.label_].append(span)

        ents, discarded = filter_spans(list(doc.ents) + matches, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
