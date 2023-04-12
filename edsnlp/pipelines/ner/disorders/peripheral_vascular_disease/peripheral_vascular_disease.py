"""`eds.peripheral_vascular_disease` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class PeripheralVascularDisease(DisorderMatcher):
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="peripheral_vascular_disease",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:
            if span._.source == "ischemia":
                if "peripheral" not in span._.assigned.keys():
                    continue

            yield span
