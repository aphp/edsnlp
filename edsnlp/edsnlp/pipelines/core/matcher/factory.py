from typing import Any, Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.pipelines.core.matcher.matcher import GenericTermMatcher
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher=GenericTermMatcher.exact,
    term_matcher_config={},
)


@deprecated_factory(
    "matcher",
    "eds.matcher",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.matcher", default_config=DEFAULT_CONFIG, assigns=["doc.ents", "doc.spans"]
)
def create_component(
    nlp: Language,
    name: str = "eds.matcher",
    terms: Optional[Dict[str, Union[str, List[str]]]] = None,
    attr: Union[str, Dict[str, str]] = None,
    regex: Optional[Dict[str, Union[str, List[str]]]] = "TEXT",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: GenericTermMatcher = GenericTermMatcher.exact,
    term_matcher_config: Dict[str, Any] = {},
):
    """
    Provides a generic matcher component.

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    name: str
        The name of the component.
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

        You won't be able to match on newlines if this is enabled and
        the "spaces"/"newline" option of `eds.normalizer` is enabled (by default).
    term_matcher: GenericTermMatcher
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher class
    """
    assert not (terms is None and regex is None)

    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return GenericMatcher(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
