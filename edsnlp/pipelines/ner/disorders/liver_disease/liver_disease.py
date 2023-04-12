"""`eds.liver_disease` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class LiverDisease(DisorderMatcher):
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="liver_disease",
            patterns=patterns,
            status_mapping={
                0: "ABSENT",
                1: "MILD",
                2: "MODERATE_TO_SEVERE",
            },
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:
            if span._.source in {"moderate_severe", "transplant"}:
                span._.status = 2

            yield span
