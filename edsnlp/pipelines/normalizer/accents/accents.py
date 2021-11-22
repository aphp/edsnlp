from typing import List, Tuple

from spacy.tokens import Doc

from ..utils import replace


class Accents(object):
    def __init__(self, accents: List[Tuple[str, str]]) -> None:
        self.accents = accents

    def __call__(self, doc: Doc) -> Doc:
        """
        Remove accents from ``normalization`` custom attribute.

        Parameters
        ----------
        doc : Doc
            The Spacy ``Doc`` object.

        Returns
        -------
        Doc
            The document, with accents removed in ``Token._.normalization``.
        """

        for token in doc:
            token._.normalization = replace(
                text=token._.normalization, rep=self.accents
            )

        return doc
