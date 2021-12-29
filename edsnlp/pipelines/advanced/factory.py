from typing import Any, Dict

from spacy.language import Language

from edsnlp.pipelines.advanced import AdvancedRegex


@Language.factory("advanced-regex")
def create_component(
    nlp: Language,
    name: str,
    regex_config: Dict[str, Any],
    window: int = 10,
    verbose: int = 0,
    ignore_excluded: bool = False,
    attr: str = "NORM",
):

    return AdvancedRegex(
        nlp,
        regex_config=regex_config,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
        attr=attr,
    )
