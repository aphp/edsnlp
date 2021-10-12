import re
from typing import List, Optional, Union

from loguru import logger
from spacy.tokens import Doc, Span


class RegexMatcher(object):
    """
    Simple RegExp matcher.

    Parameters
    ----------
    alignment_mode:
        How spans should be aligned with tokens.
        Possible values are `strict` (character indices must be aligned
        with token boundaries), "contract" (span of all tokens completely
        within the character span), "expand" (span of all tokens at least
        partially covered by the character span).
        Defaults to `expand`.
    attr: str
        Default attribute to match on, by default "TEXT".
        Can be overiden in the ``add`` method.
    """

    def __init__(
        self,
        alignment_mode: str = "expand",
        attr: str = "TEXT",
    ):
        self.alignment_mode = alignment_mode
        self.regex = dict()
        self.attr = dict()

        self.default_attr = attr

    def add(
        self,
        key: str,
        patterns: List[str],
        attr: Optional[str] = None,
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

        if not attr:
            attr = self.default_attr

        assert attr in ["TEXT", "NORM", "LOWER"]
        self.regex[key] = [re.compile(pattern) for pattern in patterns]
        self.attr[key] = attr

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
        del self.regex[key]

    def create_span(
        self,
        doclike: Union[Doc, Span],
        start: int,
        end: int,
        key: str,
    ) -> Span:
        """
        Spacy only allows strict alignment mode for char_span on Spans.
        This method circumvents this.

        Parameters
        ----------
        doclike:
            Doc or Span.
        start:
            Character index within the Doc-like object.
        end:
            Character index of the end, within the Doc-like object.
        key:
            The key used to match.

        Returns
        -------
        span:
            A span matched on the Doc-like object.
        """
        if isinstance(doclike, Doc):
            span = doclike.char_span(
                start,
                end,
                label=key,
                alignment_mode=self.alignment_mode,
            )
        else:
            span = doclike.doc.char_span(
                doclike.start_char + start,
                doclike.start_char + end,
                label=key,
                alignment_mode=self.alignment_mode,
            )

        return span

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
        -------
        span:
            A match.
        """
        normalized_text = (
            doclike[:]._.norm
            if any([self.attr[k] == "NORM" for k in self.regex])
            else None
        )

        for key, patterns in self.regex.items():
            if self.attr[key] == "NORM":
                text = normalized_text
            elif self.attr[key] == "LOWER":
                text = doclike.text.lower()
            else:
                text = doclike.text
            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.trace(f"Matched a regex from {key}: {repr(match.group())}")
                    span = self.create_span(
                        doclike,
                        match.start(),
                        match.end(),
                        key,
                    )
                    if span is not None:
                        yield span

    def __call__(
        self,
        doclike: Union[Doc, Span],
        as_spans=True,
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
        for match in self.match(doclike):
            if not as_spans:
                match = (match.label, match.start, match.end)
            yield match
