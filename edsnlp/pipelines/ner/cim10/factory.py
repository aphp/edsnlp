from typing import Any, Dict

from spacy.language import Language
from typing_extensions import Literal

from edsnlp.pipelines.core.terminology.terminology import TerminologyMatcher

from ...base import SpanSetterArg
from .patterns import get_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config={},
    label="cim10",
    span_setter={"ents": True, "cim10": True},
)


@Language.factory(
    "eds.cim10",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.cim10",
    *,
    attr: str = "NORM",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: Literal["exact", "simstring"] = "exact",
    term_matcher_config: Dict[str, Any] = {},
    label: str = "cim10",
    span_setter: SpanSetterArg = {"ents": True, "cim10": True},
):
    """
    The `eds.cim10` pipeline component extract terms from documents using the CIM10
    (French-language ICD) terminology as a reference.

    !!! warning "Very low recall"

        When using the `exact` matching mode, this component has a very poor recall
        performance. We can use the `simstring` mode to retrieve approximate matches,
        albeit at the cost of a significantly higher computation time.

    Examples
    --------
    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.cim10", config=dict(term_matcher="simstring"))

    text = "Le patient est suivi pour fièvres typhoïde et paratyphoïde."

    doc = nlp(text)

    doc.ents
    # Out: (fièvres typhoïde et paratyphoïde,)

    ent = doc.ents[0]

    ent.label_
    # Out: cim10

    ent.kb_id_
    # Out: A01
    ```

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : str
        The name of the component
    attr : str
        The default attribute to use for matching.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    ignore_space_tokens : bool
        Whether to skip space tokens during matching.
    term_matcher: TerminologyTermMatcher
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher term matcher
    label : str
        Label name to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Returns
    -------
    TerminologyMatcher

    Authors and citation
    --------------------
    The `eds.cim10` pipeline was developed by AP-HP's Data Science team.
    """
    return TerminologyMatcher(
        nlp=nlp,
        name=name,
        regex=dict(),
        terms=get_patterns(),
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
        label=label,
        span_setter=span_setter,
    )
