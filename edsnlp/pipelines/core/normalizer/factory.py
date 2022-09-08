from typing import Any, Dict, Union

from spacy import registry
from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .accents.factory import DEFAULT_CONFIG as accents_config
from .normalizer import Normalizer
from .pollution.factory import DEFAULT_CONFIG as pollution_config
from .quotes.factory import DEFAULT_CONFIG as quotes_config

DEFAULT_CONFIG = dict(
    accents=True,
    lowercase=True,
    quotes=True,
    pollution=True,
)


@deprecated_factory(
    "normalizer",
    "eds.normalizer",
    default_config=DEFAULT_CONFIG,
    assigns=["token.norm", "token.tag"],
)
@Language.factory(
    "eds.normalizer", default_config=DEFAULT_CONFIG, assigns=["token.norm", "token.tag"]
)
def create_component(
    nlp: Language,
    name: str,
    accents: Union[bool, Dict[str, Any]],
    lowercase: Union[bool, Dict[str, Any]],
    quotes: Union[bool, Dict[str, Any]],
    pollution: Union[bool, Dict[str, Any]],
):

    if accents:
        config = dict(**accents_config)
        if isinstance(accents, dict):
            config.update(accents)
        accents = registry.get("factories", "eds.accents")(nlp, "eds.accents", **config)

    if quotes:
        config = dict(**quotes_config)
        if isinstance(quotes, dict):
            config.update(quotes)
        quotes = registry.get("factories", "eds.quotes")(nlp, "eds.quotes", **config)

    if pollution:
        config = dict(**pollution_config["pollution"])
        if isinstance(pollution, dict):
            config.update(pollution)
        pollution = registry.get("factories", "eds.pollution")(
            nlp, "eds.pollution", pollution=config
        )

    normalizer = Normalizer(
        lowercase=lowercase,
        accents=accents or None,
        quotes=quotes or None,
        pollution=pollution or None,
    )

    return normalizer
