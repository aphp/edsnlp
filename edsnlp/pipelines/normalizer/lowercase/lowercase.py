from spacy.tokens import Doc

from ..utils import first_normalization


class Lowercase(object):
    def __call__(self, doc: Doc) -> Doc:
        """
        Remove accents from norm attribute

        Parameters
        ----------
        doc : Doc
            [description]

        Returns
        -------
        Doc
            [description]
        """

        for token in doc:
            first_normalization(token=token)
            token._.normalization = token._.normalization.lower()

        return doc
