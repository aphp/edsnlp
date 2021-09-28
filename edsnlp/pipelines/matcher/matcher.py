from typing import List, Dict, Optional, Any, Union

from loguru import logger

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spaczz.matcher import FuzzyMatcher

from edsnlp.base import BaseComponent
from edsnlp.matchers.regex import RegexMatcher

from edsnlp.utils.filter import filter_spans

TERM_ATTR = "term_attr"
DEFAULT_ATTR = "NORM"


class GenericMatcher(BaseComponent):
    def __init__(
        self,
        nlp: Language,
        terms: Optional[Dict[str, Union[List[str], str]]],
        attr: Union[Dict[str, str], str],
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy: bool,
        fuzzy_kwargs: Optional[Dict[str, Any]],
        filter_matches: bool,
        on_ents_only: bool,
    ):
        """
        Provides a matcher matcher component.

        Parameters
        ----------
        nlp : Language
            The Spacy object.
        terms : Optional[Dict[str, Union[List[str], str]]]
            A dictionary of terms.
        attr : Union[Dict[str, str], str]
            The attribute to use for matching.
        regex : Optional[Dict[str, Union[List[str], str]]]
            A dictionary of regular expressions.
        fuzzy : bool
            Whether to do fuzzy matching.
        fuzzy_kwargs : Optional[Dict[str, Any]]
            Default options for the fuzzy matcher.
        filter_matches : bool
            Whether to filter out matches.
        on_ents_only : bool
            Whether to to look for matches around pre-extracted entities only.
        """

        self.nlp = nlp
        self.on_ents_only = on_ents_only
        self.terms = self._to_dict_of_lists(terms)
        self.regex = self._to_dict_of_lists(regex)
        self.fuzzy = fuzzy
        self.filter_matches = filter_matches

        self.attr = self._prepare_attr(attr, self.regex, nlp.pipe_names)

        self.matcher = self._create_matcher(fuzzy, fuzzy_kwargs, self.attr[TERM_ATTR])
        self.regex_matcher = RegexMatcher()

        self._build_patterns()
        self.DEFAULT_ATTR = DEFAULT_ATTR

    def _create_matcher(self, fuzzy, fuzzy_kwargs, term_attr):
        if fuzzy:
            logger.warning(
                "You have requested fuzzy matching, which significantly increases "
                "compute times (x60 increases are common)."
            )
            if fuzzy_kwargs is None:
                fuzzy_kwargs = {"min_r2": 90, "ignore_case": True}
            return FuzzyMatcher(self.nlp.vocab, attr=term_attr, **fuzzy_kwargs)
        else:
            return PhraseMatcher(self.nlp.vocab, attr=term_attr)

    def _prepare_attr(self, attr, regex, pipe_names):
        if isinstance(attr, str):
            # Setting the provided attribute for every term/regex
            attr = {k: attr.upper() for k in set(regex) | {TERM_ATTR}}
            return attr

        attr = {k: v.upper() for k, v in attr.items()}
        for k in set(regex) | {TERM_ATTR}:
            if k not in attr:
                attr[k] = DEFAULT_ATTR

        # Checks
        diff = set(attr) - set(regex) - {TERM_ATTR}
        if diff:
            logger.warning(
                f"some of 'attr' keys are not in 'regex' keys and will be ignored: {diff}"
            )

        vals = {attr[k] for k in regex}
        if vals - {"NORM", "TEXT"}:
            raise ValueError(f"Some attributes in 'attr' are not supported: {vals}")

        vals.add(attr[TERM_ATTR])
        if "NORM" in vals and ("normalizer" not in pipe_names):
            logger.warning("You are using the NORM attribute but no normalizer is set.")

        return attr

    def _build_patterns(self):
        for key, expressions in self.terms.items():
            patterns = list(self.nlp.tokenizer.pipe(expressions))
            self.matcher.add(key, patterns)

        for key, patterns in self.regex.items():
            self.regex_matcher.add(key, patterns, self.attr[key])

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

        ents, discarded = filter_spans(list(doc.ents) + spans, return_discarded=True)

        doc.ents = ents
        doc.spans["discarded"] = discarded

        return doc

    def _to_dict_of_lists(self, d):
        d = d or dict()
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = [v]
        return d
