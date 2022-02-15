from typing import List, Tuple

from spacy.language import Language

from .patterns import quotes_and_apostrophes
from .quotes import Quotes

default_config = dict(
    quotes=quotes_and_apostrophes,
)


# noinspection PyUnusedLocal
@Language.factory("quotes", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    quotes: List[Tuple[str, str]],
):
    return Quotes(
        quotes=quotes,
    )
