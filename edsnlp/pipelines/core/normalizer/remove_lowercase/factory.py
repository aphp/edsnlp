from spacy.language import Language
from spacy.tokens import Doc

from edsnlp.utils.deprecation import deprecated_factory


def remove_lowercase(doc: Doc):
    """
    Add case on the `NORM` custom attribute. Should always be applied first.

    Parameters
    ----------
    doc : Doc
        The spaCy `Doc` object.

    Returns
    -------
    Doc
        The document, with case put back in `NORM`.
    """

    for token in doc:
        token.norm_ = token.text

    return doc


@deprecated_factory("remove-lowercase", "eds.remove_lowercase", assigns=["token.norm"])
@deprecated_factory(
    "eds.remove-lowercase", "eds.remove_lowercase", assigns=["token.norm"]
)
@Language.factory("eds.remove_lowercase", assigns=["token.norm"])
def create_component(
    nlp: Language,
    name: str,
):
    """
    Add case on the `NORM` custom attribute. Should always be applied first.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    name : str
        The name of the component.
    """
    return remove_lowercase  # pragma: no cover
