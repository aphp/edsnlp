"""`eds.cerebrovascular_accident` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class CerebrovascularAccident(DisorderMatcher):
    def __init__(self, nlp, name, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name=name,
            label_name="cerebrovascular_accident",
            patterns=patterns,
            include_assigned=False,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:
            if (span._.source == "with_localization") and (
                "brain_localized" not in span._.assigned.keys()
            ):
                continue

            if span._.source == "ischemia":
                if "brain" not in span._.assigned.keys():
                    continue

            yield span
