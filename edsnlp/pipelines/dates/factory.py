from typing import List

from spacy.language import Language

from . import Dates, terms

default_config = dict(
    dates=terms.dates,
)


# noinspection PyUnusedLocal
@Language.factory("dates", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    dates: List[str],
):
    return Dates(
        nlp,
        dates=dates,
    )
