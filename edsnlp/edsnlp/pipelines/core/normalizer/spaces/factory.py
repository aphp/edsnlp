from spacy.language import Language

from .spaces import Spaces

DEFAULT_CONFIG = dict()


@Language.factory(
    "eds.spaces",
    default_config=DEFAULT_CONFIG,
    assigns=["token.tag"],
)
def create_component(
    nlp: Language,
    name: str,
    newline: bool = True,
):
    """
    Create a new component to update the `tag_` attribute of tokens.

    We assign "SPACE" to `token.tag` to be used by optimized components
    such as the EDSPhraseMatcher

    Parameters
    ----------
    newline : bool
        Whether to update the newline tokens too
    """
    return Spaces(newline=newline)
