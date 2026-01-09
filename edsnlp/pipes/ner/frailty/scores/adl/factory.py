from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference, severity_assigner_equals_reference
from .patterns import default_patterns


@registry.factory.register(
    "eds.adl",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "adl",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(admissible_references=[6], default_reference=6),
    attr: str = "NORM",
    label: str = "adl",
    span_setter: SpanSetterArg = {"ents": True, "adl": True, "autonomy": True},
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = severity_assigner_equals_reference,
):
    """The 'eds.adl' component extracts the
    [ADL score](https://www.msdmanuals.com/fr/professional/multimedia/table/%C3%A9chelle-modifi%C3%A9e-des-activit%C3%A9s-quotidiennes-de-katz)
    """  # noqa E501
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
