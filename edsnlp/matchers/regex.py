import re
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from spacy.tokens import Doc, Span


class RegexMatcher(object):
    """
    Simple RegExp matcher.

    Parameters
    ----------
    alignment_mode:
        How spans should be aligned with tokens.
        Possible values are ``strict`` (character indices must be aligned
        with token boundaries), "contract" (span of all tokens completely
        within the character span), "expand" (span of all tokens at least
        partially covered by the character span).
        Defaults to ``expand``.
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
        self.regex = []

        self.default_attr = attr

    def add(
        self,
        key: str,
        patterns: List[str],
        attr: Optional[str] = None,
        alignment_mode: Optional[str] = None,
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
            Attribute to use for matching.
            By default uses the ``default_attr`` attribute
        """

        attr = attr or self.default_attr
        alignment_mode = alignment_mode or self.alignment_mode

        assert attr in ["TEXT", "NORM", "CUSTOM_NORM", "LOWER"]

        patterns = [re.compile(pattern) for pattern in patterns]

        self.regex.append((key, patterns, attr, alignment_mode))

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
        self.regex = [(k, p, a, am) for k, p, a, am in self.regex if k != key]

    def create_span(
        self,
        doclike: Union[Doc, Span],
        start: int,
        end: int,
        key: str,
        alignment_mode: str,
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
                alignment_mode=alignment_mode,
            )
        else:
            span = doclike.doc.char_span(
                doclike.start_char + start,
                doclike.start_char + end,
                label=key,
                alignment_mode=alignment_mode,
            )

        return span

    def match(
        self,
        doclike: Union[Doc, Span],
    ) -> Tuple[Span, re.Match]:
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

        if isinstance(doclike, Span):
            doc = doclike.doc
        else:
            doc = doclike

        for key, patterns, attr, alignment_mode in self.regex:
            if attr == "CUSTOM_NORM":
                text = doclike._.normalized.text
            elif attr == "LOWER":
                text = doclike.text.lower()
            else:
                text = doclike.text

            doclike_ = doclike._.normalized if attr == "CUSTOM_NORM" else doclike

            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.trace(f"Matched a regex from {key}: {repr(match.group())}")

                    span = self.create_span(
                        doclike=doclike_,
                        start=match.start(),
                        end=match.end(),
                        key=key,
                        alignment_mode=alignment_mode,
                    )

                    if span is None:
                        continue

                    if attr == "CUSTOM_NORM":
                        # Going back to the original document
                        start, end = span.start, span.end

                        start = doc._.norm2original[start]
                        end = doc._.norm2original[end]

                        span = Span(doc, start, end, label=span.label)

                    yield span, match

    def __call__(
        self,
        doclike: Union[Doc, Span],
        as_spans=True,
        return_groupdict=False,
    ) -> Union[Span, Tuple[Span, Dict[str, Any]]]:
        """
        Performs matching. Yields matches.

        Parameters
        ----------
        doclike:
            Spacy Doc or Span object.
        as_spans:
            Returns matches as spans.

        Yields
        ------
        span:
            A match.
        groupdict:
            Additional information coming from the named patterns
            in the regular expression.
        """
        for span, match in self.match(doclike):
            if not as_spans:
                span = (span.label, span.start, span.end)
            if return_groupdict:
                yield span, match.groupdict()
            else:
                yield span
