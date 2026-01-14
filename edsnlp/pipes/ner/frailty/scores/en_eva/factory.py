from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference
from .patterns import default_patterns


def en_eva_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    reference = ent._.assigned.get("reference", None)
    assert reference is not None, (
        "ent should have a reference not None set in _.assigned"
    )
    if reference == 10:
        return "healthy" if value < 4 else "altered_nondescript"
    elif reference == 100:
        return "healthy" if value < 40 else "altered_nondescript"
    else:
        raise ValueError(f"Unknown reference for en : {reference}")


@registry.factory.register(
    "eds.en_eva",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "en_eva",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(
        admissible_references=[100, 10], default_reference=100
    ),
    attr: str = "ORTH",
    label: str = "en_eva",
    span_setter: SpanSetterArg = {"ents": True, "en_eva": True, "pain": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = en_eva_severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="pain",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
