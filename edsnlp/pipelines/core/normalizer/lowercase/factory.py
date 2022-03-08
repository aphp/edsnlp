from spacy.language import Language
from spacy.tokens import Doc


@Language.component("remove-lowercase")
@Language.component("eds.remove-lowercase")
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
