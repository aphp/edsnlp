from spacy.tokens import Doc, Token


class Normalizer(object):
    """
    Gathers the normalisation and creates the ``normalized`` custom attributes.
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
            else:
                if Token.has_extension("end_line"):
                    if (
                        token._.end_line is False
                    ):  # I want to enter only if end_line==False, (not when end_line is None)
                        if len(spaces) > 0:
                            spaces[-1] = True

        normalized = Doc(vocab=doc.vocab, words=words, spaces=spaces)

        doc._.normalized = normalized

        return doc


class NormalizerPopulate(object):
    """
    Pipeline that populates the ``keep`` and ``normalization`` custom attributes.
    """

    def __call__(self, doc: Doc) -> Doc:
        """
        Populates ``Token._.normalization`` with ``token.text``.

        Parameters
        ----------
        doc:
            Spacy Doc object.

        Returns
        -------
        doc:
            Same document, with a modified ``normalization`` attribute for each token.
        """

        for token in doc:
            token._.normalization = token.text

        return doc
