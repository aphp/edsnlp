from typing import List, Optional, Tuple

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseComponent

from .patterns import quotes_and_apostrophes


class QuotesConverter(BaseComponent):
    """
    We normalise quotes, following this
    `source <https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html>`_.

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline object.
    name : Optional[str]
        The component name.
    quotes : List[Tuple[str, str]]
        List of quotation characters and their transcription.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: Optional[str] = "spaces",
        *,
        quotes: List[Tuple[str, str]] = quotes_and_apostrophes,
    ):
        super().__init__(nlp=nlp, name=name)
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
