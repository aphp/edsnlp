"""`eds.solid_tumor` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher
from edsnlp.pipelines.ner.scores.tnm import TNM
from edsnlp.utils.numbers import parse_digit

from .patterns import default_patterns


class SolidTumor(DisorderMatcher):
    def __init__(self, nlp, patterns, use_tnm):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="solid_tumor",
            patterns=patterns,
            detailled_statusmapping={
                0: "ABSENT",
                1: "LOCALIZED",
                2: "METASTASIS",
            },
        )

        self.use_tnm = use_tnm

        if use_tnm:
            self.tnm = TNM(nlp, pattern=None, attr="TEXT")

    def process_tnm(self, doc):
        spans = self.tnm.process(doc)
        spans = self.tnm.parse(spans)

        for span in spans:
            span.label_ = "solid_tumor"
            span._.source = "tnm"
            metastasis = span._.value.dict().get("metastasis", "0")
            if metastasis == "1":
                span._.status = 2
            yield span

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

        if self.use_tnm:
            yield from self.process_tnm(doc)
