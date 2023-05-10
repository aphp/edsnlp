from spacy.tokens import Doc


class Spaces(object):
    """
    We assign "SPACE" to `token.tag` to be used by optimized components
    such as the EDSPhraseMatcher

    Parameters
    ----------
    newline : bool
        Whether to update the newline tokens too
    """

    def __init__(self, newline: bool) -> None:
        self.newline = newline

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the component to the doc.

        Parameters
        ----------
        doc: Doc

        Returns
        -------
        doc: Doc
        """
        space_hash = doc.vocab.strings["SPACE"]
        for token in doc:
            if len(token.text.strip()) == 0:
                token.tag = space_hash

        return doc
