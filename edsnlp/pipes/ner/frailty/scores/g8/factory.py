import re
from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.matchers.utils import get_text
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import float_regex, int_regex
from .patterns import default_patterns


def score_normalization(span):
    possible_values = re.findall(
        rf"g\s?8|/|({float_regex})(?:\s*/\s*({int_regex}))?",
        get_text(span, attr="NORM", ignore_excluded=False),
    )
    kept_value = None
    kept_reference = None
    to_keep = False

    for value, reference in possible_values:
        if value == "":
            continue
        val = float(value.replace(",", "."))
        try:
            ref = int(reference)
            if ref == 17 and val <= ref:
                kept_value, kept_reference = val, ref
                to_keep = True
                break
        except ValueError:
            if (kept_reference is None) and (
                (kept_value is None) or (kept_value < val)
            ):
                kept_value, kept_reference = (val, None)

    if kept_value is None:
        return

    if (kept_reference is None) and (kept_value <= 17):
        to_keep = True
        kept_reference = 17

    if to_keep:
        span._.assigned["value"] = kept_value
        span._.assigned["reference"] = kept_reference
        return span


def g8_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    if value <= 14:
        return "altered_nondescript"
    else:
        return "healthy"


@registry.factory.register(
    "eds.g8",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "g8",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = score_normalization,
    attr: str = "NORM",
    label: str = "g8_score",
    span_setter: SpanSetterArg = {"ents": True, "g8": True, "g8_score": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = g8_severity_assigner,
):
    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="g8",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
