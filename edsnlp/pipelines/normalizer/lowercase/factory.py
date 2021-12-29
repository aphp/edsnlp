from spacy.language import Language
from spacy.tokens import Doc


# noinspection PyUnusedLocal
@Language.component("remove-lowercase")
def create_component(doc: Doc):
    """
    Add case on the ``NORM`` custom attribute. Should always be applied first.

    Parameters
    ----------
    doc : Doc
        The Spacy ``Doc`` object.

    Returns
    -------
    Doc
        The document, with case put back in ``NORM``.
    """

    for token in doc:
        token.norm_ = token.text

    return doc
