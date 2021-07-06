import re
from typing import List, Union, Iterable

from loguru import logger
from spacy.tokens import Span, Doc


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
    """

    def __init__(self, alignment_mode: str = 'expand'):
        self.alignment_mode = alignment_mode
        self.regex = dict()

    def add(self, key: str, patterns: List[str]):
        self.regex[key] = [re.compile(pattern) for pattern in patterns]

    def remove(self, key: str):
        del self.regex[key]

    def create_span(self, doclike: Union[Doc, Span], start: int, end: int, key: str) -> Span:
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

    def match(self, doclike: Union[Doc, Span]) -> Span:
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
        for key, patterns in self.regex.items():
            for pattern in patterns:
                for match in pattern.finditer(doclike.text):
                    logger.trace(f'Matched a regex from {key}: {repr(match.group())}')
                    span = self.create_span(
                        doclike,
                        match.start(),
                        match.end(),
                        key,
                    )
                    if span is not None:
                        yield span

    def __call__(self, doclike: Union[Doc, Span], as_spans=True) -> Span:
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
