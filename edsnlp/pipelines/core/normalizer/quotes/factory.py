from typing import List, Optional, Tuple

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .quotes import Quotes

DEFAULT_CONFIG = dict(
    quotes=None,
)


@deprecated_factory(
    "quotes", "eds.quotes", default_config=DEFAULT_CONFIG, assigns=["token.norm"]
)
@Language.factory(
    "eds.quotes",
    default_config=DEFAULT_CONFIG,
    assigns=["token.norm"],
)
def create_component(
    nlp: Language,
    name: str,
    quotes: Optional[List[Tuple[str, str]]],
):
    return Quotes(
        quotes=quotes,
    )
