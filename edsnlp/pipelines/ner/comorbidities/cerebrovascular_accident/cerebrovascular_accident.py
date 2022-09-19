"""`eds.comorbidities.cerebrovascular_accident` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

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

            if (span._.source == "with_localization") and (
                "brain_localized" not in span._.assigned.keys()
            ):
                continue

            if (span._.source == "AIT") and span[-1].nbor().is_upper:
                # Proxy: Then AIT is part of a name
                continue

            yield span
