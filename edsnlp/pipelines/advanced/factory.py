from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.advanced import AdvancedRegex


@Language.factory("advanced_regex")
def create_adv_regex_component(
    nlp: Language,
    name: str,
    regex_config: Dict[str, Any],
    window: int,
):

    return AdvancedRegex(nlp, regex_config, window)
