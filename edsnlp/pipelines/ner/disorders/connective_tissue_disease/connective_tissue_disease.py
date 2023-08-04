"""`eds.connective_tissue_disease` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class ConnectiveTissueDisease(DisorderMatcher):
    def __init__(self, nlp, name, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name=name,
            label_name="connective_tissue_disease",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if span._.source == "lupus" and all(tok.is_upper for tok in span):
                # Huge change of FP / Title section
                continue

            yield span
