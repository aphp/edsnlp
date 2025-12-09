from typing import Iterator, Optional

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher.models import FullConfig
from edsnlp.pipes.ner.disorders.base import DisorderMatcher

from .utils import normalize_space_characters


class FrailtyDomainMatcher(DisorderMatcher):
    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        domain: str,
        patterns: FullConfig,
        *,
        name: str = None,
        label: str = None,
        normalize_spaces: bool = True,
        span_setter: SpanSetterArg = None,
    ):
        if name is None:
            name = domain
        if label is None:
            label = domain
        if normalize_spaces:
            patterns = normalize_space_characters(patterns)
        self.domain = domain

        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            detailed_status_mapping={
                1: "other",
                2: "healthy",
                3: "altered_nondescript",
                4: "altered_mild",
                5: "altered_severe",
            },
            label=label,
            span_setter=span_setter,
            include_assigned=True,
            ignore_space_tokens=False,
        )
        self.reverse_detailed_status_mapping = dict(
            map(reversed, self.detailed_status_mapping.items())
        )

    def set_extensions(self):
        super().set_extensions()
        if not Span.has_extension(self.domain):
            Span.set_extension(self.domain, default=None)

    def process(self, doc: Doc) -> Iterator[Span]:
        for span in super().process(doc):
            status = 1
            # Assign keys
            if any("severe" in key for key in span._.assigned):
                status = 5
            elif any("mild" in key for key in span._.assigned):
                status = 4
            elif any("altered" in key for key in span._.assigned):
                status = 3
            elif any("healthy" in key for key in span._.assigned):
                status = 2
            # Regex source
            elif "healthy" in span._.source:
                status = 2
            elif "altered" in span._.source:
                status = 3
            elif "mild" in span._.source:
                status = 4
            elif "severe" in span._.source:
                status = 5
            span._.set(self.domain, self.detailed_status_mapping[status])
            yield span
