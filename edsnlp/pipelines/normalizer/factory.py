from spacy.language import Language

from edsnlp.pipelines.normalizer import Normalizer

# noinspection PyUnusedLocal
@Language.factory("normalizer")
def create_normaliser_component(
    nlp: Language,
    name: str,
    deaccentuate: bool = True,
    lowercase: bool = True,
):
    return Normalizer(deaccentuate=deaccentuate, lowercase=lowercase)
