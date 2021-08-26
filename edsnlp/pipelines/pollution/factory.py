from typing import Dict

from spacy.language import Language

from . import Pollution, terms

default_config = dict(
    pollution=terms.pollution,
)


# noinspection PyUnusedLocal
@Language.factory("pollution", default_config=default_config)
def create_pollution_component(
    nlp: Language,
    name: str,
    pollution: Dict[str, str],
):
    return Pollution(nlp, pollution=pollution)
