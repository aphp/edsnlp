from typing import List, Optional, Tuple

from spacy.tokens import Doc

from .patterns import quotes_and_apostrophes


class Quotes(object):
    """
    We normalise quotes, following this
    `source <https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html>`_.

    Parameters
    ----------
    quotes : List[Tuple[str, str]]
        List of quotation characters and their transcription.
    """

    def __init__(self, quotes: Optional[List[Tuple[str, str]]]) -> None:
        if quotes is None:
            quotes = quotes_and_apostrophes

        self.translation_table = str.maketrans(
            "".join(quote_group for quote_group, _ in quotes),
            "".join(rep * len(quote_group) for quote_group, rep in quotes),
        )

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
            token.norm_ = token.norm_.translate(self.translation_table)

        return doc
