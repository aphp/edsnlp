"""`eds.diabetes` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.core.contextual_matcher.contextual_matcher import get_window
from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import COMPLICATIONS, default_patterns


class Diabetes(DisorderMatcher):
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="diabetes",
            patterns=patterns,
            detailled_statusmapping={
                0: "ABSENT",
                1: "WITHOUT_COMPLICATION",
                2: "WITH_COMPLICATION",
            },
        )

        self.complication_matcher = RegexMatcher(
            attr="NORM", ignore_excluded=True, alignment_mode="expand"
        )
        self.complication_matcher.build_patterns(
            regex=dict(far_complications=COMPLICATIONS)
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:
            if span._.source == "complicated":
                span._.status = 2

            elif any([k.startswith("complicated") for k in span._.assigned.keys()]):
                span._.status = 2

            elif (
                get_text(span, "NORM", ignore_excluded=True) == "db"
            ) and not span._.assigned:
                # Mostly FP
                continue

            elif self.has_far_complications(span):
                span._.status = 2

            yield span

    def has_far_complications(self, span: Span):
        """
        Handles the common case where complications are listed as bullet points,
        sometimes fairly far from the anchor.
        """
        window = (0, 50)
        context = get_window(span, window, limit_to_sentence=False)
        if next(self.complication_matcher(context), None) is not None:
            return True
        return False
