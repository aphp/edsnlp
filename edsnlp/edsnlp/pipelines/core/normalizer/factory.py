from typing import Any, Dict, Union

from spacy import registry
from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .accents.factory import DEFAULT_CONFIG as accents_config
from .normalizer import Normalizer
from .pollution.factory import DEFAULT_CONFIG as pollution_config
from .quotes.factory import DEFAULT_CONFIG as quotes_config
from .spaces.factory import DEFAULT_CONFIG as spaces_config

DEFAULT_CONFIG = dict(
    accents=True,
    lowercase=True,
    quotes=True,
    spaces=True,
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
    name: str = "eds.normalizer",
    accents: Union[bool, Dict[str, Any]] = True,
    lowercase: Union[bool, Dict[str, Any]] = True,
    quotes: Union[bool, Dict[str, Any]] = True,
    spaces: Union[bool, Dict[str, Any]] = True,
    pollution: Union[bool, Dict[str, Any]] = True,
) -> Normalizer:
    """
    Normalisation pipeline. Modifies the `NORM` attribute,
    acting on five dimensions :

    - `lowercase`: using the default `NORM`
    - `accents`: deterministic and fixed-length normalisation of accents.
    - `quotes`: deterministic and fixed-length normalisation of quotation marks.
    - `spaces`: "removal" of spaces tokens (via the tag_ attribute).
    - `pollution`: "removal" of pollutions (via the tag_ attribute).

    Parameters
    ----------
    lowercase : bool
        Whether to remove case.
    accents : Union[bool, Dict[str, Any]]
        `Accents` configuration object
    quotes : Union[bool, Dict[str, Any]]
        `Quotes` configuration object
    spaces : Union[bool, Dict[str, Any]]
        `Spaces` configuration object
    pollution : Union[bool, Dict[str, Any]]
        Optional `Pollution` configuration object.
    """

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

    if spaces:
        config = dict(**spaces_config)
        if isinstance(spaces, dict):
            config.update(spaces)
        spaces = registry.get("factories", "eds.spaces")(nlp, "eds.spaces", **config)

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
        spaces=spaces or None,
    )

    return normalizer
