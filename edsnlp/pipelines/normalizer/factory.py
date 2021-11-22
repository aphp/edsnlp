from typing import Any, Dict, Union

from spacy.language import Language

from .normalizer import Normalizer, NormalizerPopulate


@Language.factory("normalizer-populate")
def create_population_component(
    nlp: Language,
    name: str,
):
    return NormalizerPopulate()


# noinspection PyUnusedLocal
@Language.factory("normalizer")
def create_component(
    nlp: Language,
    name: str,
    accents: Union[bool, Dict[str, Any]] = True,
    lowercase: Union[bool, Dict[str, Any]] = True,
    quotes: Union[bool, Dict[str, Any]] = True,
    pollution: Union[bool, Dict[str, Any]] = True,
    endlines: Union[bool, Dict[str, Any]] = False,
):

    nlp.add_pipe("normalizer-populate")

    if lowercase:
        nlp.add_pipe("lowercase")

    if accents:
        if isinstance(accents, dict):
            nlp.add_pipe("accents", config=accents)
        else:
            nlp.add_pipe("accents")

    if quotes:
        if isinstance(quotes, dict):
            nlp.add_pipe("quotes", config=quotes)
        else:
            nlp.add_pipe("quotes")

    if pollution:
        if isinstance(pollution, dict):
            nlp.add_pipe("pollution", config=pollution)
        else:
            nlp.add_pipe("pollution")

    if endlines:
        if isinstance(endlines, dict):
            nlp.add_pipe("endlines", config=endlines)
        else:
            nlp.add_pipe("endlines")

    return Normalizer()
