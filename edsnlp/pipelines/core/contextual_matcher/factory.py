from ctypes import Union
from typing import Any, Dict, List

from spacy.language import Language

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    window=10,
    verbose=0,
    ignore_excluded=False,
    attr="NORM",
)


@deprecated_factory(
    "contextual-matcher", "eds.contextual-matcher", default_config=DEFAULT_CONFIG
)
@Language.factory("eds.contextual-matcher", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
    alignment_mode: str,
    attr: str,
    ignore_excluded: bool,
):

    return ContextualMatcher(
        nlp,
        name,
        patterns,
        alignment_mode,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
