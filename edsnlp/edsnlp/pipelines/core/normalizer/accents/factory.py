from typing import List, Optional, Tuple

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .accents import Accents

DEFAULT_CONFIG = dict(
    accents=None,
)


@deprecated_factory(
    "accents", "eds.accents", default_config=DEFAULT_CONFIG, assigns=["token.norm"]
)
@Language.factory(
    "eds.accents",
    default_config=DEFAULT_CONFIG,
    assigns=["token.norm"],
)
def create_component(
    nlp: Language,
    name: str,
    accents: Optional[List[Tuple[str, str]]],
):
    return Accents(
        accents=accents,
    )
