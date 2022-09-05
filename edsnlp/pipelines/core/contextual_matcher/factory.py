import re
from typing import Any, Dict, List, Union

from spacy.language import Language

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
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
    regex_flags: Union[re.RegexFlag, int],
    include_assigned: bool,
):

    return ContextualMatcher(
        nlp,
        name,
        patterns,
        assign_as_span,
        alignment_mode,
        attr=attr,
        ignore_excluded=ignore_excluded,
        regex_flags=regex_flags,
        include_assigned=include_assigned,
    )
