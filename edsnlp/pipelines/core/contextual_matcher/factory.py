from typing import Any, Dict

from spacy.language import Language

from edsnlp.pipelines.core.advanced import AdvancedRegex
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    window=10,
    verbose=0,
    ignore_excluded=False,
    attr="NORM",
)


@deprecated_factory(
    "advanced-regex", "eds.advanced-regex", default_config=DEFAULT_CONFIG
)
@Language.factory("eds.advanced-regex", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    regex_config: Dict[str, Any],
    window: int,
    verbose: int,
    ignore_excluded: bool,
    attr: str,
):

    return AdvancedRegex(
        nlp,
        regex_config=regex_config,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
        attr=attr,
    )
