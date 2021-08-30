from spacy.tokens import Doc, Span

from .terms import quotes_and_apostrophes, accents
from typing import List, Tuple


def _get_span_norm(span: Span):
    # add the spaces between the tokens when there is one, unless for the last one.
    # from spaCy's implementation https://github.com/explosion/spaCy/blob/master/spacy/tokens/span.pyx#L509-L514
    text = "".join([x.norm_ + x.whitespace_ for x in span])
    if len(span) > 0 and span[-1].whitespace_:
        text = text[:-1]
    return text


if not Span.has_extension("norm"):
    Span.set_extension("norm", getter=_get_span_norm)


def replace(
    text: str,
    rep: List[Tuple[str, str]],
) -> str:
    """
    Replaces a list of characters in a given text.

    Parameters
    ----------
    text : str
        Text to modify.
    rep : List[Tuple[str, str]]
        List of `(old, new)` tuples. `old` can list multiple characters.

    Returns
    -------
    str
        Processed text.
    """

    for olds, new in rep:
        for old in olds:
            text = text.replace(old, new)
    return text


class Normalizer(object):
    """
    Pipeline that populates the NORM attribute.
    The goal is to handle accents without changing the document's length, thus
    keeping a 1-to-1 correspondance between raw and normalized characters.

    We also normalise quotes, following this [source](https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html).

    Parameters
    ----------
    deaccentuate:
        Whether to deaccentuate the tokens.
    lowercase:
        Whether to transform the tokens to lowercase.
    """

    def __init__(
        self,
        remove_accents: bool = True,
        lowercase: bool = False,
    ):

        self.remove_accents = remove_accents
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

        for token in doc:
            # Remove case
            s = token.lower_ if self.lowercase else token.text

            # Remove accents
            if self.remove_accents:
                s = replace(text=s, rep=accents)

            # Replace quotes and apostrophes.
            s = replace(text=s, rep=quotes_and_apostrophes)

            token.norm_ = s

        return doc
