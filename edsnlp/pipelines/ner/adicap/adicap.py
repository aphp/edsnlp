"""`eds.adicap` pipeline"""

from typing import List

from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import patterns


class Adicap(BaseComponent):
    def __init__(self, nlp, pattern, attr):

        self.nlp = nlp

        if pattern is None:
            pattern = patterns.adicap_pattern

        if isinstance(pattern, str):
            pattern = [pattern]

        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")
        self.regex_matcher.add("adicap", pattern)

        self.set_extensions()

    def process(self, doc: Doc) -> List[Span]:
        """
        Find ADICAP mentions in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        spans:
            list of ADICAP spans
        """

        spans = self.regex_matcher(
            doc,
            as_spans=True,
            return_groupdict=False,
        )

        spans = filter_spans(spans)

        return spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags ADICAP mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for ADICAP
        """
        spans = self.process(doc)
        spans = filter_spans(spans)

        # spans = self.parse(spans)

        doc.spans["adicap"] = spans

        ents, discarded = filter_spans(list(doc.ents) + spans, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
