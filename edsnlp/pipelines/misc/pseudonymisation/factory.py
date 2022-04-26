from spacy.language import Language

from .pseudonymisation import Pseudonymisation

DEFAULT_CONFIG = dict(
    attr="NORM",
)


@Language.factory("eds.pseudonymisation", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
):
    return Pseudonymisation(
        nlp,
        attr=attr,
    )
