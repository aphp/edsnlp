"""`eds.comorbidities.diabetes` pipeline"""
from operator import itemgetter
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.pipelines.qualifiers.base import get_qualifier_extensions
from edsnlp.utils.extensions import rgetattr
from edsnlp.utils.filter import filter_spans


class Comorbidity(ContextualMatcher):
    def __init__(self, nlp, name, patterns, include_assigned=True):

        self.nlp = nlp

        super().__init__(
            nlp=nlp,
            name=name,
            attr="NORM",
            patterns=patterns,
            ignore_excluded=True,
            regex_flags=0,
            alignment_mode="expand",
            assign_as_span=True,
            include_assigned=include_assigned,
        )

        if not Span.has_extension("status"):
            Span.set_extension("status", default=1)
        if not Doc.has_extension("comorbidities"):
            Doc.set_extension("comorbidities", default={})

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags entities.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            annotated spaCy Doc object
        """
        spans = self.postprocess(doc, self.process(doc))
        spans = filter_spans(spans)
        doc.spans[self.name] = spans

        self.aggregate(doc)

        return doc

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        """
        Can be overrid
        """
        yield from spans

    def aggregate(self, doc):
        """
        Aggregate extractions. Rules are:
        - Only extractions not tagged by any Qualifier are kept
        - The "worst" status is kept
        """
        spans = doc.spans[self.name]
        qualifiers = get_qualifier_extensions(self.nlp)
        kept_spans = [
            (span, span._.status)
            for span in spans
            if not any(
                [
                    rgetattr(span, qualifier_extension)
                    for qualifier_extension in qualifiers.values()
                ]
            )
        ]
        if not kept_spans:
            status = 0
        else:
            status = max(kept_spans, key=itemgetter(1))[1]

        doc._.comorbidities[self.name] = status

        return doc
