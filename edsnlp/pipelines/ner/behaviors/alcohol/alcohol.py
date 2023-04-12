"""`eds.alcohol` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class Alcohol(DisorderMatcher):
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="alcohol",
            patterns=patterns,
            status_mapping={0: "ABSENT", 1: "PRESENT", 2: "ABSTINENCE"},
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:
            if "stopped" in span._.assigned.keys():
                span._.status = 2

            elif "zero_after" in span._.assigned.keys():
                span._.status = 0

            yield span
