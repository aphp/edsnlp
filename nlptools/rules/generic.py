from typing import List, Dict, Optional, Any

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spaczz.matcher import FuzzyMatcher

from nlptools.rules.regex import RegexMatcher

from nlptools.rules.base import BaseComponent


class GenericMatcher(BaseComponent):
    """
    Provides a generic matcher component.
    """

    def __init__(
            self,
            nlp: Language,
            terms: Optional[Dict[str, List[str]]] = None,
            regex: Optional[Dict[str, List[str]]] = None,
            fuzzy: Optional[bool] = False,
            fuzzy_kwargs: Optional[Dict[str, Any]] = None,
            filter_matches: Optional[bool] = True,
    ):
        """
        Initialises the pipeline.

        Parameters
        ----------
        nlp: The Spacy object.
        terms: A dictionary of terms to look for.
        regex: A dictionary of regex patterns.
        fuzzy: Whether to perform fuzzy matching on the terms.
        fuzzy_kwargs: Default options for the fuzzy matcher, if used.
        filter_matches: Whether to filter out matches.
        """

        self.nlp = nlp

        self.terms = terms or dict()
        self.regex = regex or dict()

        self.fuzzy = fuzzy

        self.filter_matches = filter_matches

        if fuzzy:
            if fuzzy_kwargs is None:
                fuzzy_kwargs = {"min_r2": 90, "ignore_case": True}
            self.matcher = FuzzyMatcher(self.nlp.vocab, attr='LOWER', **fuzzy_kwargs)
        else:
            self.matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')

        self.regex_matcher = RegexMatcher()

        self.build_patterns()

    def build_patterns(self) -> None:
        for key, expressions in self.terms.items():
            patterns = list(self.nlp.tokenizer.pipe(expressions))
            self.matcher.add(key, patterns)

        for key, patterns in self.regex.items():
            self.regex_matcher.add(key, patterns)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc and filter out duplicates and inclusions

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        sections: List of Spans referring to sections.
        """
        matches = self.matcher(doc)
        regex_matches = self.regex_matcher(doc)

        spans = []

        for match in matches:
            match_id, start, end = match[:3]
            if not self.fuzzy:
                match_id = self.nlp.vocab.strings[match_id]
            span = Span(doc, start, end, label=match_id)
            spans.append(span)

        for match in regex_matches:
            spans.append(match)

        if self.filter_matches:
            spans = self._filter_matches(spans)

        return spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc: spaCy Doc object
        
        Returns
        -------
        doc: spaCy Doc object, annotated for extracted terms.
        """
        spans = self.process(doc)

        doc.ents = spans

        return doc
