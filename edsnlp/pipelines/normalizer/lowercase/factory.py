from spacy.language import Language

from .lowercase import Lowercase


# noinspection PyUnusedLocal
@Language.factory("lowercase")
def create_component(
    nlp: Language,
    name: str,
):
    return Lowercase()
