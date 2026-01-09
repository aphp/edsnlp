from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference, make_severity_assigner_threshold
from .patterns import default_patterns

severity_assigner = make_severity_assigner_threshold(
    threshold=1,
    healthy="high",
    comparison="strict",
)


@registry.factory.register(
    "eds.walk_speed",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "walk_speed",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(
        admissible_references=[200], default_reference=200
    ),  # TODO : better when no ref
    attr: str = "NORM",
    label: str = "walk_speed",
    span_setter: SpanSetterArg = {"ents": True, "walk_speed": True, "mobility": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="mobility",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
