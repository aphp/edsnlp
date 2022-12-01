"""`eds.comorbidities.diabetes` pipeline"""
from operator import itemgetter
from typing import Generator

from importlib_metadata import re
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.pipelines.qualifiers.base import get_qualifier_extensions
from edsnlp.utils.extensions import rgetattr
from edsnlp.utils.filter import filter_spans, sent_is_title


class Comorbidity(ContextualMatcher):
    def __init__(
        self,
        nlp,
        name,
        patterns,
        include_assigned=True,
        titles_as_hypothesis_threshold=0.8,
        aggregate_per_document=True,
    ):

        self.nlp = nlp

        super().__init__(
            nlp=nlp,
            name=name,
            attr="NORM",
            patterns=patterns,
            ignore_excluded=True,
            regex_flags=re.S,
            alignment_mode="expand",
            assign_as_span=True,
            include_assigned=include_assigned,
        )

        self.set_extensions()
        self.titles_as_hypothesis_threshold = titles_as_hypothesis_threshold
        self.aggregate_per_document = aggregate_per_document

    @classmethod
    def set_extensions(cl) -> None:

        super().set_extensions()

        if not Span.has_extension("status"):
            Span.set_extension("status", default=1)
        if not Doc.has_extension("comorbidities"):
            Doc.set_extension("comorbidities", default={})
        # if not Span.has_extension("title_ratio"):
        #     Span.set_extension("title_ratio", default=9)

        for qualifier in ["negation", "family", "hypothesis"]:
            if not Token.has_extension(qualifier):
                Token.set_extension(qualifier, default=False)
            if not Span.has_extension(qualifier):
                Span.set_extension(qualifier, default=False)

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
        if self.titles_as_hypothesis_threshold is not None:
            spans = self.set_titles_as_hypothesis(spans)
        spans = filter_spans(spans)

        doc.spans[self.name] = spans

        if self.aggregate_per_document:
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

    def set_titles_as_hypothesis(self, spans: Generator[Span, None, None]):
        """
        Method determine if an entity is in a
        - Title
        - Sidenote / Footnote
        - Information paragraph

        We simply check if more than half of the tokens
        in the sentence starts with an uppercase

        Parameters
        ----------
        ent : Span
            An entity
        """

        for ent in spans:
            title_ratio = sent_is_title(ent.sent, neighbours=False)
            if (ent[0].is_upper or ent[0].is_title) and (
                title_ratio >= self.titles_as_hypothesis_threshold
            ):
                ent._.hypothesis = True
            yield ent
