from typing import Any, Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher, TerminologyTermMatcher

DEFAULT_CONFIG = dict(
    terms=None,
    attr="TEXT",
    regex=None,
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config={},
)


@Language.factory(
    "eds.terminology",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    label: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    name: str = "eds.terminology",
    attr: Union[str, Dict[str, str]] = "TEXT",
    regex: Optional[Dict[str, Union[str, List[str]]]] = None,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: TerminologyTermMatcher = "exact",
    term_matcher_config: Dict[str, Any] = {},
):
    """
    Provides a terminology matching component.

    The terminology matching component differs from the simple matcher component in that
    the `regex` and `terms` keys are used as spaCy's `kb_id`. All matched entities
    have the same label, defined in the top-level constructor (argument `label`).

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    name: str
        The name of the component.
    label : str
        Top-level label
    terms : Optional[Patterns]
        A dictionary of terms.
    regex : Optional[Patterns]
        A dictionary of regular expressions.
    attr : str
        The default attribute to use for matching.
        Can be overridden using the `terms` and `regex` configurations.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    ignore_space_tokens: bool
        Whether to skip space tokens during matching.
    term_matcher: TerminologyTermMatcher
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher class
    """
    assert not (terms is None and regex is None)

    return TerminologyMatcher(
        nlp,
        label=label,
        terms=terms or dict(),
        attr=attr,
        regex=regex or dict(),
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
