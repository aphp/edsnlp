from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.advanced import AdvancedRegex


@Language.factory("advanced_regex")
def create_component(
    nlp: Language,
    name: str,
    regex_config: Dict[str, Any],
    window: int = 10,
    verbose: int = 0,
):

    return AdvancedRegex(
        nlp,
        regex_config=regex_config,
        window=window,
        verbose=verbose,
    )
