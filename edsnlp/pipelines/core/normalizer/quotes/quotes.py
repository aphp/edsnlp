from typing import List, Tuple

from spacy.tokens import Doc

from ..utils import replace


class Quotes(object):
    """
    We normalise quotes, following this
    `source <https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html>`_.

    Parameters
    ----------
    quotes : List[Tuple[str, str]]
        List of quotation characters and their transcription.
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
            token.norm_ = replace(text=token.norm_, rep=self.quotes)

        return doc
