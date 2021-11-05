from typing import Union

from spacy.tokens import Doc, Span


class Normalizer(object):
    """
    Pipeline that populates the ``NORM`` attribute.
    The goal is to handle accents without changing the document's length, thus
    keeping a 1-to-1 correspondance between raw and normalized characters.

    We also normalise quotes, following this `source <https://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html>`_.
    """

    def __call__(self, doc: Doc) -> Doc:
        """
        Normalises the document. Creates a normalised version
        of the document, excluding polluted tokens.

        Parameters
        ----------
        doc:
            Spacy Doc object.

        Returns
        -------
        doc:
            Same document, with a modified NORM attribute for each token.
        """

        words = []
        spaces = []

        for token in doc:
            if token._.keep:
                words.append(token._.normalization)
                spaces.append(bool(token.whitespace_))

        normalized = Doc(vocab=doc.vocab, words=words, spaces=spaces)

        doc._.normalized = normalized

        return doc
