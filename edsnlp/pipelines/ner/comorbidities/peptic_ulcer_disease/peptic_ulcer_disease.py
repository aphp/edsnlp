"""`eds.comorbidities.peptic_ulcer_disease` pipeline"""

from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class PepticUlcerDisease(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="peptic_ulcer_disease",
            patterns=patterns,
            include_assigned=False,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if (span._.source == "generic") and not span._.assigned:
                continue

            yield span
