from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns


def iadl_severity_assigner(ent: Span):
    """We know that 0 is severe, and a perfect score is normal.
    Beyond that, we know it is altered, but there is no validated severity gradation."""

    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    reference = ent._.assigned.get("reference", None)
    assert reference is not None, (
        "ent should have a reference not None set in _.assigned"
    )
    if value == reference:
        return "healthy"
    elif value == 0:
        return "altered_severe"
    else:
        return "altered_nondescript"


@registry.factory.register(
    "eds.iadl",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "iadl",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(
        admissible_references=[4, 5, 8, 11], default_reference=4
    ),
    attr: str = "NORM",
    label: str = "iadl",
    span_setter: SpanSetterArg = {"ents": True, "iadl": True, "autonomy": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = iadl_severity_assigner,
):
    """The 'eds.iadl' component extracts the IADL score.
    Nowadays, the IADL score is mostly evaluated on 4 items, but it also exists
    with 5, 8 or 11 items."""
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="autonomy",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
