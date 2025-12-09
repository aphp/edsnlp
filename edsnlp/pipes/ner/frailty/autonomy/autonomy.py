from typing import Optional

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher.models import FullConfig

from ..base import FrailtyDomainMatcher
from ..utils import normalize_space_characters
from .patterns import default_patterns


class AutonomyMatcher(FrailtyDomainMatcher):
    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        *,
        name: str = "autonomy",
        patterns: FullConfig = default_patterns,
        label: str = "autonomy",
        normalize_spaces: bool = True,
        span_setter: SpanSetterArg = {"ents": True, "autonomy": True},
    ):
        if normalize_spaces:
            patterns = normalize_space_characters(patterns)
        super().__init__(
            nlp=nlp,
            domain="autonomy",
            patterns=patterns,
            name=name,
            label=label,
            span_setter=span_setter,
        )
