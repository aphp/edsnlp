from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns


def tug_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"

    if value < 20:
        return "healthy"
    elif value >= 30:
        return "altered_severe"
    else:
        return "altered_mild"


@registry.factory.register(
    "eds.tug",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "tug",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(
        admissible_references=[200], default_reference=200
    ),  # TODO : better when no ref
    attr: str = "NORM",
    label: str = "tug",
    span_setter: SpanSetterArg = {"ents": True, "tug": True, "mobility": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = tug_severity_assigner,
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
