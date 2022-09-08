"""`eds.comorbidities.cerebrovascular_accident` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity
from edsnlp.utils.filter import filter_spans

from .patterns import default_patterns


class CerebrovascularAccident(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="cerebrovascular_accident",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if span._.source == "complicated":
                span._.status = 2

            elif any([k.startswith("complicated") for k in span._.assigned.keys()]):
                span._.status = 2

            yield span
