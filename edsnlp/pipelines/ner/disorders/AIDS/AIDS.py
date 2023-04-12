"""`eds.AIDS` pipeline"""
import itertools
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.disorders.base import DisorderMatcher
from edsnlp.pipelines.qualifiers.hypothesis import Hypothesis
from edsnlp.pipelines.qualifiers.hypothesis.factory import (
    DEFAULT_CONFIG as DEFAULT_CONFIG_HYP,
)
from edsnlp.pipelines.qualifiers.negation import Negation
from edsnlp.pipelines.qualifiers.negation.factory import (
    DEFAULT_CONFIG as DEFAULT_CONFIG_NEG,
)

from .patterns import default_patterns


class AIDS(DisorderMatcher):
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="AIDS",
            patterns=patterns,
            include_assigned=False,
        )

        DEFAULT_CONFIG_NEG.update({"on_ents_only": "AIDS_opportunist"})
        DEFAULT_CONFIG_HYP.update({"on_ents_only": "AIDS_opportunist"})

        self.inner_negation = Negation(
            nlp,
            **DEFAULT_CONFIG_NEG,
        )

        self.inner_hypothesis = Hypothesis(
            nlp,
            **DEFAULT_CONFIG_HYP,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        spans = list(spans)

        doc.spans["AIDS_opportunist"] = list(
            itertools.chain.from_iterable(
                [span._.assigned.get("opportunist", []) for span in spans]
            )
        )

        doc = self.inner_negation(
            self.inner_hypothesis(
                doc,
            )
        )

        for span in spans:
            opportunists = span._.assigned.get("opportunist", [])
            if opportunists:
                opportunists = [
                    ent
                    for ent in opportunists
                    if not (ent._.negation or ent._.hypothesis)
                ]
            stage = "stage" in span._.assigned

            if span._.source == "hiv" and not (opportunists or stage):
                continue

            yield span

        del doc.spans["AIDS_opportunist"]
