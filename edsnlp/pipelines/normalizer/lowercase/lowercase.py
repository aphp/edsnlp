from spacy.tokens import Doc


class Lowercase(object):
    def __call__(self, doc: Doc) -> Doc:
        """
        Remove case from ``normalization`` custom attribute.

        Parameters
        ----------
        doc : Doc
            The Spacy ``Doc`` object.

        Returns
        -------
        Doc
            The document, with case removed in ``Token._.normalization``.
        """

        for token in doc:
            token._.normalization = token._.normalization.lower()

        return doc
