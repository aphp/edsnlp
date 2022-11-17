from enum import Enum
from itertools import chain
from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.simstring import SimstringMatcher
from edsnlp.matchers.utils import Patterns
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans


class TerminologyTermMatcher(str, Enum):
    exact = "exact"
    simstring = "simstring"


class TerminologyMatcher(BaseComponent):
    """
    Provides a terminology matching component.

    The terminology matching component differs from the simple matcher component in that
    the `regex` and `terms` keys are used as spaCy's `kb_id`. All matched entities
    have the same label, defined in the top-level constructor (argument `label`).

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    label : str
        Top-level label
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
    term_matcher: TerminologyTermMatcher
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher class
    """

    def __init__(
        self,
        nlp: Language,
        label: str,
        terms: Optional[Patterns],
        regex: Optional[Patterns],
        attr: str,
        ignore_excluded: bool,
        term_matcher: TerminologyTermMatcher = TerminologyTermMatcher.exact,
        term_matcher_config=None,
    ):

        self.nlp = nlp

        self.label = label

        self.attr = attr

        if term_matcher == TerminologyTermMatcher.exact:
            self.phrase_matcher = EDSPhraseMatcher(
                self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                **(term_matcher_config or {}),
            )
        elif term_matcher == TerminologyTermMatcher.simstring:
            self.phrase_matcher = SimstringMatcher(
                vocab=self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                **(term_matcher_config or {}),
            )
        else:
            raise ValueError(
                f"Algorithm {repr(term_matcher)} does not belong to"
                f" known matchers [exact, simstring]."
            )

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms)
        self.regex_matcher.build_patterns(regex=regex)

        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Post-process matches to account for terminology.

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

        spans = []

        for match in chain(matches, regex_matches):
            span = Span(
                doc=match.doc,
                start=match.start,
                end=match.end,
                label=self.label,
                kb_id=match.label,
            )
            span._.set(self.label, match.label_)
            spans.append(span)

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

        if self.label not in doc.spans:
            doc.spans[self.label] = matches

        ents, discarded = filter_spans(list(doc.ents) + matches, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
