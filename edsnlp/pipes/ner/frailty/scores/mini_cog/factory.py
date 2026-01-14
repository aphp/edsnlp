from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns

score_normalization = make_find_value_and_reference(
    admissible_references=[5], default_reference=5
)


def minicog_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    if value <= 2:
        return "altered_nondescript"
    else:
        return "healthy"


@registry.factory.register(
    "eds.mini_cog",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "mini_cog",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = score_normalization,
    attr: str = "NORM",
    label: str = "mini_cog",
    span_setter: SpanSetterArg = {"ents": True, "mini_cog": True, "cognition": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = minicog_severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="cognition",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
