import re
from typing import Any, Dict, List, Union

from spacy.language import Language

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=True,
    ignore_space_tokens=False,
    regex_flags=0,
    alignment_mode="expand",
    assign_as_span=False,
    include_assigned=False,
)


@deprecated_factory(
    "contextual-matcher", "eds.contextual-matcher", default_config=DEFAULT_CONFIG
)
@Language.factory("eds.contextual-matcher", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
    assign_as_span: bool,
    alignment_mode: str,
    attr: str,
    ignore_excluded: bool,
    ignore_space_tokens: bool,
    regex_flags: Union[re.RegexFlag, int],
    include_assigned: bool,
):
    """
    Allows additional matching in the surrounding context of the main match group,
    for qualification/filtering.

    Parameters
    ----------
    nlp : Language
        spaCy `Language` object.
    name : str
        The name of the pipe
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]]
        The configuration dictionary
    assign_as_span : bool
        Whether to store eventual extractions defined via the `assign` key as Spans
        or as string
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    alignment_mode : str
        Overwrite alignment mode.
    regex_flags : Union[re.RegexFlag, int]
        RegExp flags to use when matching, filtering and assigning (See
        [here](https://docs.python.org/3/library/re.html#flags))
    include_assigned : bool
        Whether to include (eventual) assign matches to the final entity

    """

    return ContextualMatcher(
        nlp,
        name,
        patterns,
        assign_as_span,
        alignment_mode,
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        regex_flags=regex_flags,
        include_assigned=include_assigned,
    )
