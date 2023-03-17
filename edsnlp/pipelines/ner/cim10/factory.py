from typing import Any, Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher, TerminologyTermMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher=TerminologyTermMatcher.exact,
    term_matcher_config={},
)


@Language.factory(
    "eds.cim10", default_config=DEFAULT_CONFIG, assigns=["doc.ents", "doc.spans"]
)
def create_component(
    nlp: Language,
    name: str = "eds.cim10",
    attr: Union[str, Dict[str, str]] = "NORM",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: TerminologyTermMatcher = TerminologyTermMatcher.exact,
    term_matcher_config: Dict[str, Any] = {},
):
    """
    Create a factory that returns new a component to recognize and normalize CIM10 codes
    and concepts in documents.

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
    ignore_space_tokens: bool
        Whether to skip space tokens during matching.
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
        label="cim10",
        regex=None,
        terms=patterns.get_patterns(),
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
