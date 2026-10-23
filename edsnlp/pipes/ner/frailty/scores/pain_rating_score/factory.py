from typing import Any, Callable, Dict, List, Optional, Union

from spacy.tokens import Span

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg

from ..base import FrailtyScoreMatcher
from ..utils import make_find_value_and_reference, value_and_reference
from .patterns import (
    allowed_uppercase_score_words,
    blacklisted_eva_context,
    blacklisted_score_text,
    blacklisted_score_word_prefixes,
    blacklisted_score_words,
    default_patterns,
    pain_cue,
    pain_template,
    simple_en_score,
)


def is_blacklisted_score(ent: Span) -> bool:
    """
    Reject a normalized EN, EVA or EVN score whose context shows another scale,
    e.g. `EVA fatigue 8/10`, or an uppercase heading, e.g. `EN CAS DE DOULEURS : 5`.
    Since EN is also a French preposition, we keep it only with a pain cue, a pain
    template before it or a plain score shape. `normalize_and_filter_score` calls it
    after the configured normalization.
    """
    value = ent._.assigned["value"]
    score_name = next(
        (token for token in ent if token.text.upper() in {"EN", "EVA", "EVN"}),
        None,
    )
    if score_name is None:
        return False

    if blacklisted_score_text.search(ent.text):
        return True

    for match in value_and_reference.finditer(ent.text):
        if float(match.group(1).replace(",", ".")) == value:
            value_match_start = match.start(1)
            break
    else:
        return False

    tokens_between_score_and_value = [
        token
        for token in ent
        if score_name.idx + len(score_name)
        <= token.idx
        < ent.start_char + value_match_start
    ]
    if any(
        token.text.isupper()
        and sum(char.isalpha() for char in token.text) > 1
        and token.text not in allowed_uppercase_score_words
        for token in tokens_between_score_and_value
    ):
        return True

    left_context = ent.doc[max(0, ent.start - 8) : ent.start].text
    if score_name.text.upper() == "EN":
        if pain_cue.search(ent.text):
            return False
        if pain_template.search(left_context):
            return False
        if simple_en_score.match(ent.text):
            return False
        return True

    if blacklisted_eva_context.search(left_context) and not pain_cue.search(ent.text):
        return True

    return any(
        token.norm_.lower().strip("'’") in blacklisted_score_words
        or token.norm_.lower()
        .strip("'’")
        .startswith(tuple(blacklisted_score_word_prefixes))
        for token in tokens_between_score_and_value
    )


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
    "eds.pain_rating_score",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "pain_rating_score",
    *,
    patterns: List[Dict] = default_patterns,
    score_normalization: Union[
        str, Callable[[Span], Any]
    ] = make_find_value_and_reference(
        admissible_references=[100, 10], default_reference=10
    ),
    attr: str = "ORTH",
    label: str = "pain_rating_score",
    span_setter: SpanSetterArg = {
        "ents": True,
        "pain_rating_score": True,
        "pain_status": True,
    },
    include_assigned: bool = True,
    severity_assigner: Callable[[Span], Any] = en_eva_severity_assigner,
):
    """
    The `eds.pain_rating_score` component extracts mentions of
    the EN (Echelle Numérique), EVN (Echelle Verbale Numérique) and EVA
    (Echelle Visuelle Analogique) scores.

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
    if isinstance(score_normalization, str):
        score_normalization = registry.misc.get(score_normalization)

    def normalize_and_filter_score(ent: Span):
        """
        Apply the configured normalization then reject scores from other scales
        """
        ent = score_normalization(ent)
        if ent is None or is_blacklisted_score(ent):
            return None
        return ent

    return FrailtyScoreMatcher(
        nlp,
        name=name,
        patterns=patterns,
        attr=attr,
        label=label,
        span_setter=span_setter,
        domain="pain_status",
        include_assigned=include_assigned,
        score_normalization=normalize_and_filter_score,
        severity_assigner=severity_assigner,
    )
