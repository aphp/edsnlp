from typing import List, Union

from spacy.language import Vocab
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span


class EDSPhraseMatcher(object):
    """
    Simple RegExp matcher.

    Parameters
    ----------
    attr: str
        Default attribute to match on, by default "TEXT".
        Can be overiden in the ``add`` method.
    """

    def __init__(
        self,
        vocab: Vocab,
        attr: str = "TEXT",
    ):
        assert attr in ["TEXT", "NORM", "CUSTOM_NORM", "LOWER"]

        self.norm = attr == "CUSTOM_NORM"

        if self.norm:
            attr = "TEXT"

        self.matcher = PhraseMatcher(vocab, attr=attr)

    def add(
        self,
        key: str,
        patterns: List[str],
    ):
        """
        Add a pattern.

        Parameters
        ----------
        key : str
            Key of the new/updated pattern.
        patterns : List[str]
            List of patterns to add.
        attr : str, optional
            Attribute to use for matching, by default "TEXT"
        """

        self.matcher.add(key, patterns)

    def remove(
        self,
        key: str,
    ):
        """
        Remove a pattern.

        Parameters
        ----------
        key : str
            key of the pattern to remove.
        """
        self.matcher.remove(key)

    def match(
        self,
        doclike: Union[Doc, Span],
    ) -> Span:
        """
        Iterates on the matches.

        Parameters
        ----------
        doclike:
            Spacy Doc or Span object to match on.

        Yields
        -------s
        span:
            A match.
        """

        if isinstance(doclike, Span):
            doc = doclike.doc
        else:
            doc = doclike

        if self.norm:
            doclike = doclike._.normalized

        for label, start, end in self.matcher(doclike):
            if self.norm:
                start = doc._.norm2original[start]
                end = doc._.norm2original[end]
            yield label, start, end

    def __call__(
        self,
        doclike: Union[Doc, Span],
        as_spans=False,
    ) -> Span:
        """
        Performs matching. Yields matches.

        Parameters
        ----------
        doclike:
            Spacy Doc or Span object.
        as_spans:
            Returns matches as spans.

        Yields
        -------
        match:
            A match.
        """
        if isinstance(doclike, Span):
            doc = doclike.doc
        else:
            doc = doclike

        for label, start, end in self.match(doclike):
            if as_spans:
                match = Span(doc, start, end, label)
            else:
                match = (label, start, end)
            yield match
