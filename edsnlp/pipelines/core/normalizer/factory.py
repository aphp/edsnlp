from typing import Any, Dict, Union

from spacy import registry
from spacy.language import Language

from .accents.factory import default_config as accents_config
from .normalizer import Normalizer
from .pollution.factory import default_config as pollution_config
from .quotes.factory import default_config as quotes_config


# noinspection PyUnusedLocal
@Language.factory("normalizer")
def create_component(
    nlp: Language,
    name: str,
    accents: Union[bool, Dict[str, Any]] = True,
    lowercase: Union[bool, Dict[str, Any]] = True,
    quotes: Union[bool, Dict[str, Any]] = True,
    pollution: Union[bool, Dict[str, Any]] = True,
):

    if accents:

        config = dict(**accents_config)
        if isinstance(accents, dict):
            config.update(accents)
        accents = registry.get("factories", "accents")(nlp, "accents", **config)

    if quotes:
        config = dict(**quotes_config)
        if isinstance(quotes, dict):
            config.update(quotes)
        quotes = registry.get("factories", "quotes")(nlp, "quotes", **config)

    if pollution:
        config = dict(**pollution_config)
        if isinstance(pollution, dict):
            config.update(pollution)
        pollution = registry.get("factories", "pollution")(nlp, "pollution", **config)

    normalizer = Normalizer(
        lowercase=lowercase,
        accents=accents or None,
        quotes=quotes or None,
        pollution=pollution or None,
    )

    return normalizer
