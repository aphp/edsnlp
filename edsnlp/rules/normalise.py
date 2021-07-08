from spacy.tokens import Token, Doc
from unidecode import unidecode
from spacy import Language


class Normaliser(object):
    """
    Pipeline that populates the NORM attribute.

    Parameters
    ----------
    deaccentuate:
        Whether to deaccentuate the tokens.
    lowercase:
        Whether to transform the tokens to lowercase.
    """

    def __init__(
            self,
            deaccentuate: bool = True,
            lowercase: bool = False,
    ):

        self.deaccentuate = deaccentuate
        self.lowercase = lowercase

    def __call__(self, doc: Doc) -> Doc:
        """
        Normalises the document.

        Parameters
        ----------
        doc:
            Spacy Doc object.

        Returns
        -------
        doc:
            Same document, with a modified NORM attribute for each token.
        """
        if not (self.deaccentuate or self.lowercase):
            return doc

        for token in doc:
            if self.deaccentuate and self.lowercase:
                token.norm_ = unidecode(token.lower_)
            elif self.deaccentuate:
                token.norm_ = unidecode(token.text)
            else:
                token.norm_ = token.lower_

        return doc
