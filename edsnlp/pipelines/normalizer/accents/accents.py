from typing import List, Tuple

from spacy.tokens import Doc

from ..utils import first_normalization, replace


class Accents(object):
    def __init__(self, accents: List[Tuple[str, str]]) -> None:
        self.accents = accents

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
            first_normalization(token)
            token._.normalization = replace(
                text=token._.normalization, rep=self.accents
            )

        return doc
