"""`eds.comorbidities.CKD` pipeline"""
from typing import Generator, Optional
from loguru import logger

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class CKD(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="CKD",
            patterns=patterns,
        )

    def classify_from_dfg(self, dfg_span: Optional[Span]):
        if dfg_span is None:
            return False
        try:
            dfg_value = float(dfg_span.text.replace(",", ".").strip())
        except ValueError:
            logger.trace(f"DFG value couldn't be extracted from {dfg_span.text}")
            return False

        return dfg_value < 60  # We keep only moderate to severe CKD

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if span._.source == "dialysis" and "chronic" not in span._.assigned.keys():
                continue

            if span._.source == "general":
                if {"class", "status"} & set(span._.assigned.keys()):
                    yield span
                    continue
                elif self.classify_from_dfg(span._.assigned.get("dfg", None)):
                    yield span
                    continue
                else:
                    continue

            yield span
