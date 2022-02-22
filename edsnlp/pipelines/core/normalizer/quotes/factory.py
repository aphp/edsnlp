from typing import List, Tuple

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .patterns import quotes_and_apostrophes
from .quotes import Quotes

DEFAULT_CONFIG = dict(
    quotes=quotes_and_apostrophes,
)


@deprecated_factory("quotes", "eds.quotes", default_config=DEFAULT_CONFIG)
@Language.factory("eds.quotes", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    quotes: List[Tuple[str, str]],
):
    return Quotes(
        quotes=quotes,
    )
