import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from spacy.tokens import Doc, Span, Token

from .utils import ATTRIBUTES, Patterns, get_text


@lru_cache(maxsize=32)
def alignment(
    doclike: Union[Doc, Span],
    attr: str = "TEXT",
    ignore_excluded: bool = True,
) -> Tuple[np.array, np.array, np.array]:
    """
    Align different representations of a ``Doc`` or ``Span`` object.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        SpaCy ``Doc`` or ``Span`` object
    attr : str, optional
        Attribute to use, by default "TEXT"
    ignore_excluded : bool, optional
        Whether to remove excluded tokens, by default True

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        An alignment tuple: original, clean and offset arrays.
    """

    attr = attr.upper()

    attr = ATTRIBUTES.get(attr, attr)

    custom = attr.startswith("_")

    if custom:
        attr = attr[1:].lower()

    def length(token: Token):
        if custom:
            text = getattr(token._, attr)
        else:
            text = getattr(token, attr)
        return len(text)

    original = []
    clean = []

    cursor = 0

    for token in doclike:

        if not ignore_excluded or not token._.excluded:
            original.append(token.idx)
            clean.append(cursor)

            cursor += length(token)

            if token.whitespace_:
                original.append(token.idx + len(token.text))
                clean.append(cursor)

                cursor += 1

    clean.append(cursor)

    if ignore_excluded and token._.excluded:
        original.append(original[-1])
    else:
        original.append(token.idx + len(token.text_with_ws))

    original = np.array(original)
    clean = np.array(clean)

    offset = original - clean

    return original, clean, offset


def exclusion2original(
    doclike: Union[Doc, Span],
    attr: str,
    ignore_excluded: bool,
    index: int,
) -> int:
    """
    Goes from the cleaned text to the original one.

    Parameters
    ----------
    doc : Doc
        Doc to clean
    attr: str
        Which attribute to align.
    ignore_excluded: bool
        Whether excluded tokens should be removed
    index : int
        Index to transform

    Returns
    -------
    int:
        Converted index.
    """

    attr = attr.upper()

    if attr == "TEXT" and not ignore_excluded:
        return index

    _, exclusion, offset = alignment(
        doclike, attr=attr, ignore_excluded=ignore_excluded
    )

    arg = (exclusion >= index).argmax()
    index += offset[arg]

    return index


def get_first_included(doclike: Union[Doc, Span]) -> Token:
    for token in doclike:
        if not token._.excluded:
            return token


class RegexMatcher(object):
    """
    Simple RegExp matcher.

    Parameters
    ----------
    alignment_mode : str
        How spans should be aligned with tokens.
        Possible values are ``strict`` (character indices must be aligned
        with token boundaries), "contract" (span of all tokens completely
        within the character span), "expand" (span of all tokens at least
        partially covered by the character span).
        Defaults to ``expand``.
    attr : str
        Default attribute to match on, by default "TEXT".
        Can be overiden in the ``add`` method.
    ignore_excluded : bool
        Whether to skip exclusions
    """

    def __init__(
        self,
        alignment_mode: str = "expand",
        attr: str = "TEXT",
        ignore_excluded: bool = False,
    ):
        self.alignment_mode = alignment_mode
        self.regex = []

        self.default_attr = attr

        self.ignore_excluded = ignore_excluded

    def build_patterns(self, regex: Patterns):
        """
        Build patterns and adds them for matching.
        Helper function for pipelines using this matcher.

        Parameters
        ----------
        regex : Patterns
            Dictionary of label/terms, or label/dictionary of terms/attribute.
        """
        if not regex:
            regex = dict()

        for key, patterns in regex.items():
            if isinstance(patterns, dict):
                attr = patterns.get("attr")
                alignment_mode = patterns.get("alignment_mode")
                patterns = patterns.get("regex")
            else:
                attr = None
                alignment_mode = None

            if isinstance(patterns, str):
                patterns = [patterns]

            self.add(
                key=key, patterns=patterns, attr=attr, alignment_mode=alignment_mode
            )

    def add(
        self,
        key: str,
        patterns: List[str],
        attr: Optional[str] = None,
        ignore_excluded: Optional[bool] = None,
        alignment_mode: Optional[str] = None,
    ):
        """
        Add a pattern to the registry.

        Parameters
        ----------
        key : str
            Key of the new/updated pattern.
        patterns : List[str]
            List of patterns to add.
        attr : str, optional
            Attribute to use for matching.
            By default uses the ``default_attr`` attribute
        ignore_excluded : bool, optional
            Whether to skip excluded tokens during matching.
        alignment_mode : str, optional
            Overwrite alignment mode.
        """

        if attr is None:
            attr = self.default_attr

        if ignore_excluded is None:
            ignore_excluded = self.ignore_excluded

        if alignment_mode is None:
            alignment_mode = self.alignment_mode

        patterns = [re.compile(pattern) for pattern in patterns]

        self.regex.append((key, patterns, attr, ignore_excluded, alignment_mode))

    def remove(
        self,
        key: str,
    ):
        """
        Remove a pattern for the registry.

        Parameters
        ----------
        key : str
            key of the pattern to remove.

        Raises
        ------
        ValueError
            If the key is not present in the registered patterns.
        """
        n = len(self.regex)
        self.regex = [(k, p, a, i, am) for k, p, a, i, am in self.regex if k != key]
        if len(self.regex) == n:
            raise ValueError(f"`{key}` is not referenced in the matcher")

    def create_span(
        self,
        doclike: Union[Doc, Span],
        start: int,
        end: int,
        key: str,
        attr: str,
        alignment_mode: str,
        ignore_excluded: bool,
    ) -> Span:
        """
        SpaCy only allows strict alignment mode for char_span on Spans.
        This method circumvents this.

        Parameters
        ----------
        doclike : Union[Doc, Span]
            Doc or Span.
        start : int
            Character index within the Doc-like object.
        end : int
            Character index of the end, within the Doc-like object.
        key : str
            The key used to match.
        alignment_mode : str
            The alignment mode.
        ignore_excluded : bool
            Whether to skip excluded tokens.

        Returns
        -------
        span:
            A span matched on the Doc-like object.
        """
        doc = doclike if isinstance(doclike, Doc) else doclike.doc

        _, exclusion, offset = alignment(doc)

        if ignore_excluded:
            first_included = get_first_included(doclike)
            first, off = exclusion[first_included.i], offset[first_included.i]
        else:
            first, off = 0, 0

        start = exclusion2original(
            doc,
            attr=attr,
            ignore_excluded=ignore_excluded,
            index=start + first,
        )
        end = exclusion2original(
            doc,
            attr=attr,
            ignore_excluded=ignore_excluded,
            index=end + first,
        )
        start = start - off
        end = end - off

        start += doclike[0].idx
        end += doclike[0].idx

        span = doc.char_span(
            start,
            end,
            label=key,
            alignment_mode=alignment_mode,
        )

        return span

    def get_text(self, doclike: Union[Doc, Span], attr: str) -> str:

        return get_text(
            doclike=doclike,
            attr=attr,
            ignore_excluded=self.ignore_excluded,
        )

    def __len__(self):
        return len(set([regex[0] for regex in self.regex]))

    def match(
        self,
        doclike: Union[Doc, Span],
    ) -> Tuple[Span, re.Match]:
        """
        Iterates on the matches.

        Parameters
        ----------
        doclike:
            SpaCy Doc or Span object to match on.

        Yields
        -------
        span:
            A match.
        """

        for key, patterns, attr, ignore_excluded, alignment_mode in self.regex:
            text = self.get_text(doclike, attr)

            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.trace(f"Matched a regex from {key}: {repr(match.group())}")

                    span = self.create_span(
                        doclike=doclike,
                        start=match.start(),
                        end=match.end(),
                        key=key,
                        attr=attr,
                        alignment_mode=alignment_mode,
                        ignore_excluded=ignore_excluded,
                    )

                    if span is None:
                        continue

                    yield span, match

    def __call__(
        self,
        doclike: Union[Doc, Span],
        as_spans=False,
        return_groupdict=False,
    ) -> Union[Span, Tuple[Span, Dict[str, Any]]]:
        """
        Performs matching. Yields matches.

        Parameters
        ----------
        doclike:
            SpaCy Doc or Span object.
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
                offset = doclike[0].i
                span = (span.label, span.start - offset, span.end - offset)
            if return_groupdict:
                yield span, match.groupdict()
            else:
                yield span
