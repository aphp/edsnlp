from typing import List, Tuple

from spacy.tokens import Doc

from ..utils import first_normalization, replace


class Quotes(object):
    """
    We normalise quotes, following this `source <https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html>`_.

    Parameters
    ----------
    quotes : List[Tuple[str, str]]
        [description]
    """

    def __init__(self, quotes: List[Tuple[str, str]]) -> None:
        self.quotes = quotes

    def __call__(self, doc: Doc) -> Doc:
        """
        Normalises quotes.

        Parameters
        ----------
        doc : Doc
            Document to process.

        Returns
        -------
        Doc
            Same document, with quotes normalised.
        """

        for token in doc:
            first_normalization(token=token)
            token._.normalization = replace(text=token._.normalization, rep=self.quotes)

        return doc
