from typing import Any, Dict

from spacy.language import Language
from typing_extensions import Literal

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.core.terminology.terminology import TerminologyMatcher

from .patterns import get_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config={},
    label="drug",
    span_setter={"ents": True, "drug": True},
)


@Language.factory(
    "eds.drugs",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.drugs",
    *,
    attr: str = "NORM",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: Literal["exact", "simstring"] = "exact",
    term_matcher_config: Dict[str, Any] = {},
    label: str = "drug",
    span_setter: SpanSetterArg = {"ents": True, "drug": True},
):
    """
    The `eds.drugs` pipeline component detects mentions of French drugs (brand names and
    active ingredients) and adds them to `doc.ents`. Each drug is mapped to an
    [ATC](https://enwp.org/?curid=2770) code through the Romedi terminology
    ([@cossin:hal-02987843]). The ATC classifies drugs into groups.

    Examples
    --------
    In this example, we are looking for an oral antidiabetic medication
    (ATC code: A10B).

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.drugs", config=dict(term_matcher="exact"))

    text = "Traitement habituel: Kardégic, cardensiel (bisoprolol), glucophage, lasilix"

    doc = nlp(text)

    drugs_detected = [(x.text, x.kb_id_) for x in doc.ents]

    drugs_detected[0]
    # Out: ('Kardégic', 'B01AC06')

    len(drugs_detected)
    # Out: 5

    oral_antidiabetics_detected = list(
        filter(lambda x: (x[1].startswith("A10B")), drugs_detected)
    )
    oral_antidiabetics_detected
    # Out: [('glucophage', 'A10BA02')]
    ```

    Glucophage is the brand name of a medication that contains metformine, the
    first-line medication for the treatment of type 2 diabetes.

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
    term_matcher: Literal["exact", "simstring"]
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

    # Authors and citation

    The `eds.drugs` pipeline was developed by the IAM team and CHU de Bordeaux's Data
    Science team.
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
