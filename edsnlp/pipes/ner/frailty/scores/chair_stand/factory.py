import re
from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.matchers.utils import get_text
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import float_regex
from .patterns import default_patterns


def score_normalization(span):
    possible_values = re.findall(
        rf"5\s+levers de chaise|/|({float_regex})",
        get_text(span, attr="NORM", ignore_excluded=False),
    )
    kept_value = None
    to_keep = False

    for value in possible_values:
        if value == "":
            continue
        val = float(value.replace(",", "."))
        kept_value = val
        to_keep = True
        break

    if kept_value is None:
        return

    if to_keep:
        span._.assigned["value"] = kept_value
        return span


def chair_stand_severity_assigner(ent: Span):
    value = ent._.assigned.get("value", None)
    assert value is not None, "ent should have a value not None set in _.assigned"

    if value < 11.2:
        return "healthy"
    elif value >= 16.70:
        return "altered_severe"
    else:
        return "altered_mild"


@registry.factory.register(
    "eds.chair_stand",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "chair_stand",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = score_normalization,
    attr: str = "NORM",
    label: str = "chair_stand",
    span_setter: SpanSetterArg = {"ents": True, "chair_stand": True, "mobility": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = chair_stand_severity_assigner,
):
    """
    The `eds.chair_stand` component extracts the
    chair stand score.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : Optional[str]
        Name of the component
    patterns : List[str]
        A list of regexes to identify the score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    score_normalization : Union[str, Callable[[Union[str,None]], Any]]
        Function that takes the "raw" value extracted from the `value_extract`
        regex and should return:

        - None if no score could be extracted
        - The desired score value else
    label : str
        Label name to use for the `Span` object and the extension
    span_setter: SpanSetterArg
        How to set matches on the doc
    domain : str
        The frailty domain the score is related to
    severity_assigner: Callable[[Union[str, Tuple[float, int], Tuple[int, int]]], Any]
        Function that takes the score value and assigns the corresponding severity
        for the domain.

    Returns
    -------
    SimpleScoreMatcher
    """
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
