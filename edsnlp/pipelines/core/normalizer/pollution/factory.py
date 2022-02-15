from typing import Dict, List, Union

from spacy.language import Language

from . import Pollution, patterns

default_config = dict(
    pollution=patterns.pollution,
)


# noinspection PyUnusedLocal
@Language.factory("pollution", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    pollution: Dict[str, Union[str, List[str]]],
):
    return Pollution(
        nlp,
        pollution=pollution,
    )
