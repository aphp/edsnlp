from spacy.tokens import Token, Doc, Span
from unidecode import unidecode


def _get_span_norm(span: Span):
    # add the spaces between the tokens when there is one, unless for the last one.
    # from spaCy's implementation https://github.com/explosion/spaCy/blob/master/spacy/tokens/span.pyx#L509-L514 
    text = "".join([x.norm_ + x.whitespace_ for x in span])
    if len(span)>0 and span[-1].whitespace_:
        text = text[:-1]
    return text

if not Span.has_extension('norm'):
    Span.set_extension("norm", getter=_get_span_norm)


class Normaliser(object):
    """
    Pipeline that populates the NORM attribute.
    The goal is to handle accents without changing the document's length, thus
    keeping a 1-to-1 correspondance between raw and normalized characters.

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
        
        self.list_rep = [("ç", "c"), ("àáâä", "a"), ("èéêë", "e"), ("ìíîï", "i"), ("òóôö", "o"), ("ùúûü", "u")]
        if not self.lowercase:
            self.list_rep += [(c_in.upper(), c_out.upper()) for c_in, c_out in self.list_rep]
        

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
            s = token.lower_ if self.lowercase else token.text
            
            if self.deaccentuate:
                s = _deaccentuate_rep(self.list_rep, s)

            token.norm_ = s

        return doc

    
def _deaccentuate_rep(list_rep, s):
    for l_c_in, c_out in list_rep:
        for c_in in l_c_in:
            s = s.replace(c_in, c_out)
    return s

