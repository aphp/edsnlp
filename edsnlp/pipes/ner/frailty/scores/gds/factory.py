from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns

score_normalization = make_find_value_and_reference(
    admissible_references=[30, 15, 5, 4], default_reference=30
)


def gds_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    reference = ent._.assigned.get("reference", None)
    assert reference is not None, (
        "ent should have a reference not None set in _.assigned"
    )
    if reference == 30:
        return "altered_nondescript" if value >= 13 else "healthy"
    elif reference == 15:
        return "altered_nondescript" if value >= 5 else "healthy"
    elif reference == 5:
        return "altered_nondescript" if value >= 2 else "healthy"
    elif reference == 4:
        return "altered_nondescript" if value >= 1 else "healthy"
    else:
        raise ValueError(f"Unkown reference for GDS : {reference}")


@registry.factory.register(
    "eds.gds",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "gds",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = score_normalization,
    attr: str = "NORM",
    label: str = "gds",
    span_setter: SpanSetterArg = {"ents": True, "gds": True, "thymic": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = gds_severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="thymic",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
