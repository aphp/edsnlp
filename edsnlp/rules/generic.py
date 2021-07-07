from typing import List, Dict, Optional, Any, Union

from loguru import logger
from loguru import logger

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spaczz.matcher import FuzzyMatcher

from edsnlp.rules.base import BaseComponent
from edsnlp.rules.regex import RegexMatcher


class GenericMatcher(BaseComponent):
    """
    Provides a generic matcher component.

    Parameters
    ----------
    nlp:
        The Spacy object.
    terms:
        A dictionary of terms to look for.
    regex:
        A dictionary of regex patterns.
    fuzzy:
        Whether to perform fuzzy matching on the terms.
    fuzzy_kwargs:
        Default options for the fuzzy matcher, if used.
    filter_matches:
        Whether to filter out matches.
    on_ents_only:
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    """

    def __init__(
            self,
            nlp: Language,
            terms: Optional[Dict[str, Union[List[str], str]]] = None,
            attr: str = "TEXT",
            regex: Optional[Dict[str, Union[List[str], str]]] = None,
            fuzzy: bool = False,
            fuzzy_kwargs: Optional[Dict[str, Any]] = None,
            filter_matches: bool = True,
            on_ents_only: bool = False,
    ):

        self.nlp = nlp

        self.on_ents_only = on_ents_only

        self.terms = terms or dict()
        for k, v in self.terms.items():
            if isinstance(v, str):
                self.terms[k] = [v]

        self.regex = regex or dict()
        for k, v in self.regex.items():
            if isinstance(v, str):
                self.regex[k] = [v]

        self.fuzzy = fuzzy

        self.filter_matches = filter_matches

        if attr.upper() == "NORM" and ('normaliser' not in nlp.pipe_names):
            logger.warning("You are using the NORM attribute but no normaliser is set.")

        if fuzzy:
            logger.warning(
                'You have requested fuzzy matching, which significantly increases '
                'compute times (x60 increases are common).'
            )
            if fuzzy_kwargs is None:
                fuzzy_kwargs = {"min_r2": 90, "ignore_case": True}
            self.matcher = FuzzyMatcher(self.nlp.vocab, attr=attr, **fuzzy_kwargs)
        else:
            self.matcher = PhraseMatcher(self.nlp.vocab, attr=attr)

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
        doc:
            spaCy Doc object

        Returns
        -------
        sections:
            List of Spans referring to sections.
        """

        if self.on_ents_only:
            matches = []
            regex_matches = []

            for sent in set([ent.sent for ent in doc.ents]):
                matches += self.matcher(sent)
                regex_matches += self.regex_matcher(sent, as_spans=True)

        else:
            matches = self.matcher(doc)
            regex_matches = self.regex_matcher(doc, as_spans=True)

        spans = []

        for match in matches:
            match_id, start, end = match[:3]
            if not self.fuzzy:
                match_id = self.nlp.vocab.strings[match_id]
            span = Span(doc, start, end, label=match_id)
            spans.append(span)

        spans.extend(regex_matches)

        if self.filter_matches:
            spans = filter_spans(spans)

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
        spans = self.process(doc)

        doc.ents = spans

        return doc
