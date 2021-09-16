from spacy.language import Language

from edsnlp.pipelines.normalizer import Normalizer

# noinspection PyUnusedLocal
@Language.factory("normalizer")
def create_component(
    nlp: Language,
    name: str,
    remove_accents: bool = True,
    lowercase: bool = True,
    normalize_quotes: bool = True,
):
    return Normalizer(
        remove_accents=remove_accents,
        lowercase=lowercase,
        normalize_quotes=normalize_quotes,
    )
