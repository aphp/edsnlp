from typing import Optional

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher.models import FullConfig

from ..base import FrailtyDomainMatcher
from ..utils import normalize_space_characters
from .patterns import default_patterns


class IncontinenceMatcher(FrailtyDomainMatcher):
    """
    The `eds.incontinence` pipeline component extracts mentions of incontinence.

    Extensions
    ----------
    On each span `span` that match, the following attribute is available:

    `span._.incontinence`: set to None.
    It will specify the severity of the mention regarding the incontinence
    of the patient.
    Possible values are:
        - healthy : this span suggests the patient is well regarding that domain.
        - altered_nondescript : this span suggests the patient is not well, but
            it is not yet possible to ascertain the degree of alteration.
        - altered_mild : this span suggests a light alteration regarding
            this domain.
        - altered_severe : this span suggests a severe alteration regarding
            this domain.
        - other : this span is not indicative of the level of  alteration
            regarding this domain. Still, it hints that this domain has
            been evaluated.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(f"eds.incontinence")
    ```
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        *,
        name: str = "incontinence",
        patterns: FullConfig = default_patterns,
        label: str = "incontinence",
        normalize_spaces: bool = True,
        span_setter: SpanSetterArg = {"ents": True, "incontinence": True},
    ):
        if normalize_spaces:
            patterns = normalize_space_characters(patterns)
        super().__init__(
            nlp=nlp,
            domain="incontinence",
            patterns=patterns,
            name=name,
            label=label,
            span_setter=span_setter,
        )
