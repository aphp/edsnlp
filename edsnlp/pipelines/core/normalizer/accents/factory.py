from typing import List, Tuple

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .accents import Accents
from .patterns import accents

DEFAULT_CONFIG = dict(
    accents=accents,
)


@deprecated_factory("accents", "eds.accents", default_config=DEFAULT_CONFIG)
@Language.factory("eds.accents", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    accents: List[Tuple[str, str]],
):
    return Accents(
        accents=accents,
    )
