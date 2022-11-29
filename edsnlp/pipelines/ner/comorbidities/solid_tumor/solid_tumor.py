"""`eds.comorbidities.solid_tumor` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity
from edsnlp.utils.numbers import parse_digit

from .patterns import default_patterns


class SolidTumor(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="solid_tumor",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if (span._.source == "metastasis") or (
                "metastasis" in span._.assigned.keys()
            ):
                span._.status = 2

            if "stage" in span._.assigned.keys():
                stage = parse_digit(
                    span._.assigned["stage"],
                    atttr="NORM",
                    ignore_excluded=True,
                )
                if stage == 4:
                    span._.status = 2

            yield span
