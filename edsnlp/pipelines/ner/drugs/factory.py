from typing import Any, Dict

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher, TerminologyTermMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    term_matcher=TerminologyTermMatcher.exact,
    term_matcher_config={},
)


@Language.factory(
    "eds.drugs",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.drugs",
    attr: str = "NORM",
    ignore_excluded: bool = False,
    term_matcher: TerminologyTermMatcher = TerminologyTermMatcher.exact,
    term_matcher_config: Dict[str, Any] = {},
):
    """
    Create a new component to recognize and normalize drugs in documents.
    The terminology is based on Romedi (see documentation) and the
    drugs are normalized to the ATC codes.

    Parameters
    ----------
    nlp: Language
        spaCy `Language` object.
    name: str
        The name of the pipe
    attr: Union[str, Dict[str, str]]
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded: bool
        Whether to skip excluded tokens during matching.
    term_matcher: TerminologyTermMatcher
        The term matcher to use, either `TerminologyTermMatcher.exact` or
        `TerminologyTermMatcher.simstring`
    term_matcher_config: Dict[str, Any]
        The configuration for the term matcher

    Returns
    -------
    TerminologyMatcher
    """
    return TerminologyMatcher(
        nlp,
        label="drug",
        terms=patterns.get_patterns(),
        regex=dict(),
        attr=attr,
        ignore_excluded=ignore_excluded,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
