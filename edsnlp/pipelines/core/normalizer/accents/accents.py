from typing import List, Optional, Tuple

from spacy import Language
from spacy.tokens import Doc

from . import patterns


class AccentsConverter(object):
    """
    Normalises accents, using a same-length strategy.

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object.
    name : Optional[str]
        The component name.
    accents : List[Tuple[str, str]]
        List of accentuated characters and their transcription.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: Optional[str] = "eds.spaces",
        *,
        accents: List[Tuple[str, str]] = patterns.accents,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.translation_table = str.maketrans(
            "".join(accent_group for accent_group, _ in accents),
            "".join(rep * len(accent_group) for accent_group, rep in accents),
        )

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
            token.norm_ = token.norm_.translate(self.translation_table)

        return doc
