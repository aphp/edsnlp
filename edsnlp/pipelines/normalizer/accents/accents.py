from typing import List, Tuple

from spacy.tokens import Doc

from ..utils import replace


class Accents(object):
    """
    Normalises accents, using a same-length strategy.

    Parameters
    ----------
    accents : List[Tuple[str, str]]
        List of accentuated characters and their transcription.
    """

    def __init__(self, accents: List[Tuple[str, str]]) -> None:
        self.accents = accents

    def __call__(self, doc: Doc) -> Doc:
        """
        Remove accents from spacy ``NORM`` attribute.

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
            token.norm_ = replace(text=token.norm_, rep=self.accents)

        return doc
