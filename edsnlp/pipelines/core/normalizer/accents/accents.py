from typing import List, Optional, Tuple

from spacy.tokens import Doc

from ..utils import replace
from . import patterns


class Accents(object):
    """
    Normalises accents, using a same-length strategy.

    Parameters
    ----------
    accents : List[Tuple[str, str]]
        List of accentuated characters and their transcription.
    """

    def __init__(self, accents: Optional[List[Tuple[str, str]]]) -> None:
        if accents is None:
            accents = patterns.accents
        self.accents = accents

    def __call__(self, doc: Doc) -> Doc:
        """
        Remove accents from spacy `NORM` attribute.

        Parameters
        ----------
        doc : Doc
            The spaCy `Doc` object.

        Returns
        -------
        Doc
            The document, with accents removed in `Token.norm_`.
        """

        for token in doc:
            token.norm_ = replace(text=token.norm_, rep=self.accents)

        return doc
