from typing import Optional

from spacy.tokens import Doc

from .accents import Accents
from .lowercase import remove_lowercase
from .pollution import Pollution
from .quotes import Quotes


class Normalizer(object):
    """
    Normalisation pipeline. Modifies the `NORM` attribute,
    acting on four dimensions :

    - `lowercase`: using the default `NORM`
    - `accents`: deterministic and fixed-length normalisation of accents.
    - `quotes`: deterministic and fixed-length normalisation of quotation marks.
    - `pollution`: removal of pollutions.

    Parameters
    ----------
    lowercase : bool
        Whether to remove case.
    accents : Optional[Accents]
        Optional `Accents` object.
    quotes : Optional[Quotes]
        Optional `Quotes` object.
    pollution : Optional[Pollution]
        Optional `Pollution` object.
    """

    def __init__(
        self,
        lowercase: bool,
        accents: Optional[Accents],
        quotes: Optional[Quotes],
        pollution: Optional[Pollution],
    ):
        self.lowercase = lowercase
        self.accents = accents
        self.quotes = quotes
        self.pollution = pollution

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the normalisation pipeline, one component at a time.

        Parameters
        ----------
        doc : Doc
            spaCy `Doc` object

        Returns
        -------
        Doc
            Doc object with `NORM` attribute modified
        """
        if not self.lowercase:
            remove_lowercase(doc)
        if self.accents is not None:
            self.accents(doc)
        if self.quotes is not None:
            self.quotes(doc)
        if self.pollution is not None:
            self.pollution(doc)

        return doc
