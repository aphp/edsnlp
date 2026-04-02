from typing import Optional

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher.models import FullConfig

from ..base import FrailtyDomainMatcher
from .patterns import default_patterns


class GAMatcher(FrailtyDomainMatcher):
    """
    This pipeline component extracts mentions of geriatric assessment.
    Contrary to most of the other subclasses of FrailtyDomainMatcher, this one
    does not really match a domain per se, but explicit mentions of geriatric
    assessment itself.
    The relative rarity of those mentions motivated the development of the matchers
    for each domain, but it still can be relevant to look for them when trying to
    categorize a patient's frailty.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/frailty/geriatric_assessment/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attribute is available:

    - `span._.geriatric_assessment`: set to None.\n
    It will specify the severity of the mention regarding the geriatric assessment
    of the patient.\n
    Possible values are:

    - `healthy` : this span suggests the patient is well regarding that domain.
    - `altered_nondescript` : this span suggests the patient is not well, but
        it is not yet possible to ascertain the degree of alteration.
    - `altered_mild` : this span suggests a light alteration regarding
        this domain.
    - `altered_severe` : this span suggests a severe alteration regarding
        this domain.
    - `other` : this span is not indicative of the level of  alteration
        regarding this domain. Still, it hints that this domain has
        been evaluated.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(f"eds.geriatric_assessment")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/geriatric_assessment-examples.md"

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline
    name : Optional[str]
        The name of the component
    patterns: FullConfig
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        *,
        name: str = "geriatric_assessment",
        patterns: FullConfig = default_patterns,
        label: str = "geriatric_assessment",
        span_setter: SpanSetterArg = {"ents": True, "geriatric_assessment": True},
    ):
        super().__init__(
            nlp=nlp,
            domain="geriatric_assessment",
            patterns=patterns,
            name=name,
            label=label,
            span_setter=span_setter,
        )
