from typing import List, Tuple

from spacy.language import Language

from .accents import Accents
from .patterns import accents

default_config = dict(
    accents=accents,
)


# noinspection PyUnusedLocal
@Language.factory("accents", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    accents: List[Tuple[str, str]],
):
    return Accents(
        accents=accents,
    )
