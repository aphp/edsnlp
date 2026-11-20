from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import find_duration
from .patterns import default_patterns


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
    "eds.chair_stand_score",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "chair_stand_score",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[str, Callable[[Span], Any]] = find_duration,
    attr: str = "NORM",
    label: str = "chair_stand_score",
    span_setter: SpanSetterArg = {
        "ents": True,
        "chair_stand_score": True,
        "mobility_status": True,
    },
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = chair_stand_severity_assigner,
):
    """
    The `eds.chair_stand_score` component extracts the
    chair stand score.

    The duration is stored in seconds, e.g. `5 levers de chaise en 1 min 5 s`
    gives 65, and a number followed by another unit is ignored.

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
        domain="mobility_status",
        include_assigned=include_assigned,
        score_normalization=score_normalization,
        severity_assigner=severity_assigner,
    )
