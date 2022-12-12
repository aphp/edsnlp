"""`eds.comorbidities.tobacco` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity
from edsnlp.utils.numbers import parse_digit

from .patterns import default_patterns


class Tobacco(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="tobacco",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if "stopped" in span._.assigned.keys():
                span._.status = 2

            if "zero_after" in span._.assigned.keys():
                span._.negation = True

            if "secondhand" in span._.assigned.keys():
                span._.negation = True

            elif "PA" in span._.assigned.keys():
                pa = parse_digit(
                    span._.assigned["PA"],
                    atttr="NORM",
                    ignore_excluded=True,
                )
                if (pa == 0) and ("stopped" not in span._.assigned.keys()):
                    span._.negation = True

            yield span
