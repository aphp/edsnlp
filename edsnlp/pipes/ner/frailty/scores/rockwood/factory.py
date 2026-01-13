from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns

score_normalization = make_find_value_and_reference(
    admissible_references=[
        9,
    ],
    default_reference=9,
)


def rockwood_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    if value < 4:
        return "healthy"
    elif value >= 7:
        return "altered_severe"
    else:
        return "altered_mild"


@registry.factory.register(
    "eds.rockwood",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "rockwood",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = score_normalization,
    attr: str = "NORM",
    label: str = "rockwood",
    span_setter: SpanSetterArg = {
        "ents": True,
        "rockwood": True,
        "general_status": True,
    },
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = rockwood_severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="general_status",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
