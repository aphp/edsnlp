from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference, severity_assigner_equals_reference
from .patterns import default_patterns


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
    severity_assigner: Callable[[Span], Any] = severity_assigner_equals_reference,
):
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
