from typing import List, Union

from spacy.language import Language

from . import Dates, terms

default_config = dict(
    no_year=terms.no_year,
    absolute=terms.absolute,
    relative=terms.relative

)

# noinspection PyUnusedLocal
@Language.factory("dates", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    no_year: Union[List[str], str],
    absolute: Union[List[str], str],
    relative: Union[List[str], str]
):
    return Dates(
        nlp,
        no_year=no_year,
        absolute=absolute,
        relative=relative
    )
