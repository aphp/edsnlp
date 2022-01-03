from typing import Any, Dict, Union

from spacy.language import Language


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

    if not lowercase:
        nlp.add_pipe("remove-lowercase")

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

    return lambda doc: doc
