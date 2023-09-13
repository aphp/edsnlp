from typing import Optional

from spacy import Language
from spacy.tokens import Doc

from .accents.accents import AccentsConverter
from .pollution.pollution import PollutionTagger
from .quotes.quotes import QuotesConverter
from .remove_lowercase.factory import remove_lowercase
from .spaces.spaces import SpacesTagger


class Normalizer(object):
    """
    Normalisation pipeline. Modifies the `NORM` attribute,
    acting on five dimensions :

    - `lowercase`: using the default `NORM`
    - `accents`: deterministic and fixed-length normalisation of accents.
    - `quotes`: deterministic and fixed-length normalisation of quotation marks.
    - `spaces`: "removal" of spaces tokens (via the tag_ attribute).
    - `pollution`: "removal" of pollutions (via the tag_ attribute).

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object.
    name : Optional[str]
        The name of the component.
    lowercase : bool
        Whether to remove case.
    accents : Optional[Accents]
        Optional `Accents` object.
    quotes : Optional[Quotes]
        Optional `Quotes` object.
    spaces : Optional[Spaces]
        Optional `Spaces` object.
    pollution : Optional[Pollution]
        Optional `Pollution` object.
    """

    def __init__(
        self,
        nlp: Optional[Language],
        name: Optional[str] = "eds.normalizer",
        *,
        lowercase: bool = False,
        accents: Optional[AccentsConverter] = None,
        quotes: Optional[QuotesConverter] = None,
        spaces: Optional[SpacesTagger] = None,
        pollution: Optional[PollutionTagger] = None,
    ):
        self.nlp = nlp
        self.name = name
        self.lowercase = lowercase
        self.accents = accents
        self.quotes = quotes
        self.spaces = spaces
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
        if self.spaces is not None:
            self.spaces(doc)
        if self.pollution is not None:
            self.pollution(doc)
        return doc
