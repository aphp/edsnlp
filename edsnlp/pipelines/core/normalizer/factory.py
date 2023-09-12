from typing import Any, Dict, Union

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .accents.accents import AccentsConverter
from .normalizer import Normalizer
from .pollution.patterns import default_enabled as default_enabled_pollution
from .pollution.pollution import PollutionTagger
from .quotes.quotes import QuotesConverter
from .spaces.spaces import SpacesTagger

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
    assigns=["token.norm", "token.tag"],
)
@Language.factory("eds.normalizer", assigns=["token.norm", "token.tag"])
def create_component(
    nlp: Language,
    name: str = "eds.normalizer",
    *,
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
    nlp: Language
        The pipeline object.
    name : str
        The component name.
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
        accents = AccentsConverter(
            nlp=nlp,
            name="eds.accents",
            **(accents if accents is not True else {}),
        )

    if quotes:
        quotes = QuotesConverter(
            nlp=nlp,
            name="eds.quotes",
            **(quotes if quotes is not True else {}),
        )

    if spaces:
        spaces = SpacesTagger(
            nlp=nlp,
            name="eds.spaces",
            **(spaces if spaces is not True else {}),
        )

    if pollution:
        config = dict(default_enabled_pollution)
        if isinstance(pollution, dict):
            config.update(pollution)
        pollution = PollutionTagger(
            nlp=nlp,
            name="eds.pollution",
            pollution=config,
        )

    normalizer = Normalizer(
        nlp=nlp,
        name=name,
        lowercase=lowercase,
        accents=accents or None,
        quotes=quotes or None,
        pollution=pollution or None,
        spaces=spaces or None,
    )

    return normalizer
