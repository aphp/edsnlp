from bisect import bisect_left
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from spacy.tokens import Doc, Span, Token

from edsnlp.utils.regex import compile_regex, re

from .utils import Patterns, alignment, get_text, offset


@lru_cache(32)
def get_first_included(doclike: Union[Doc, Span]) -> Token:
    for token in doclike:
        if token.tag_ != "EXCLUDED":
            return token
    raise IndexError("The provided Span does not include any token")


def get_normalized_variant(doclike) -> str:
    tokens = [t.text + t.whitespace_ for t in doclike if not t._.excluded]
    variant = "".join(tokens)
    variant = variant.rstrip(" ")
    variant = re.sub(r"\s+", " ", variant)
    return variant


def spans_generator(match: re.Match) -> Tuple[int, int]:
    """
    Iterates over every group, and then yields the full match

    Parameters
    ----------
    match : re.Match
        A match object

    Yields
    ------
    Tuple[int, int]
        A tuple containing the start and end of the group or match
    """
    for idx in range(1, len(match.groups()) + 1):
        yield match.start(idx), match.end(idx)
    yield match.start(0), match.end(0)


def span_from_match(
    match: re.Match,
    span_from_group: bool,
) -> Tuple[int, int]:
    """
    Return the span (as a (start, end) tuple) of the first matching group.
    If `span_from_group=True`, returns the full match instead.

    Parameters
    ----------
    match : re.Match
        The Match object
    span_from_group : bool
        Whether to work on groups or on the full match

    Returns
    -------
    Tuple[int, int]
        A tuple containing the start and end of the group or match
    """
    if not span_from_group:
        start_char, end_char = match.start(), match.end()
    else:
        start_char, end_char = next(filter(lambda x: x[0] >= 0, spans_generator(match)))
    return start_char, end_char


def create_span(
    doclike: Union[Doc, Span],
    start_char: int,
    end_char: int,
    key: str,
    attr: str,
    alignment_mode: str,
    ignore_excluded: bool,
) -> Span:
    """
    spaCy only allows strict alignment mode for char_span on Spans.
    This method circumvents this.
    Parameters
    ----------
    doclike : Union[Doc, Span]
        `Doc` or `Span`.
    start_char : int
        Character index within the Doc-like object.
    end_char : int
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

    # Handle the simple case immediately
    if attr in {"TEXT", "LOWER"} and not ignore_excluded:
        off = doclike[0].idx
        return doc.char_span(
            start_char + off,
            end_char + off,
            label=key,
            alignment_mode=alignment_mode,
        )

    # If doclike is a Span, we need to get the clean
    # index of the first included token
    if ignore_excluded:
        original, clean = alignment(
            doc=doc,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        first_included = get_first_included(doclike)
        i = bisect_left(original, first_included.idx)
        first = clean[i]

    else:
        first = doclike[0].idx

    start_char = (
        first
        + start_char
        + offset(
            doc,
            attr=attr,
            ignore_excluded=ignore_excluded,
            index=first + start_char,
        )
    )

    end_char = (
        first
        + end_char
        + offset(
            doc,
            attr=attr,
            ignore_excluded=ignore_excluded,
            index=first + end_char,
        )
    )

    span = doc.char_span(
        start_char,
        end_char,
        label=key,
        alignment_mode=alignment_mode,
    )

    return span


class RegexMatcher(object):
    """
    Simple RegExp matcher.

    Parameters
    ----------
    alignment_mode : str
        How spans should be aligned with tokens.
        Possible values are `strict` (character indices must be aligned
        with token boundaries), "contract" (span of all tokens completely
        within the character span), "expand" (span of all tokens at least
        partially covered by the character span).
        Defaults to `expand`.
    attr : str
        Default attribute to match on, by default "TEXT".
        Can be overiden in the `add` method.
    flags : Union[re.RegexFlag, int]
        Additional flags provided to the `re` module.
        Can be overiden in the `add` method.
    ignore_excluded : bool
        Whether to skip exclusions
    span_from_group : bool
        If set to `False`, will create spans basede on the regex's full match.
        If set to `True`, will use the first matching capturing group as a span
        (and fall back to using the full match if no capturing group is matching)
    """

    def __init__(
        self,
        alignment_mode: str = "expand",
        attr: str = "TEXT",
        ignore_excluded: bool = False,
        flags: Union[re.RegexFlag, int] = 0,  # No additional flags
        span_from_group: bool = False,
    ):
        self.alignment_mode = alignment_mode
        self.regex = []

        self.default_attr = attr

        self.flags = flags
        self.span_from_group = span_from_group

        self.ignore_excluded = ignore_excluded

        self.set_extensions()

    @classmethod
    def set_extensions(cls):
        if not Span.has_extension("normalized_variant"):
            Span.set_extension("normalized_variant", getter=get_normalized_variant)

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
                flags = patterns.get("flags")
                patterns = patterns.get("regex")
            else:
                attr = None
                alignment_mode = None
                flags = None

            if isinstance(patterns, str):
                patterns = [patterns]

            self.add(
                key=key,
                patterns=patterns,
                attr=attr,
                alignment_mode=alignment_mode,
                flags=flags,
            )

    def add(
        self,
        key: str,
        patterns: List[str],
        attr: Optional[str] = None,
        ignore_excluded: Optional[bool] = None,
        alignment_mode: Optional[str] = None,
        flags: Optional[re.RegexFlag] = None,
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
            By default uses the `default_attr` attribute
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

        if flags is None:
            flags = self.flags

        patterns = [compile_regex(pattern, flags) for pattern in patterns]

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
            spaCy Doc or Span object to match on.

        Yields
        -------
        span:
            A match.
        """

        for key, patterns, attr, ignore_excluded, alignment_mode in self.regex:
            text = get_text(doclike, attr, ignore_excluded)

            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.trace(f"Matched a regex from {key}: {repr(match.group())}")

                    start_char, end_char = span_from_match(
                        match=match,
                        span_from_group=self.span_from_group,
                    )

                    span = create_span(
                        doclike=doclike,
                        start_char=start_char,
                        end_char=end_char,
                        key=key,
                        attr=attr,
                        alignment_mode=alignment_mode,
                        ignore_excluded=ignore_excluded,
                    )

                    if span is None:
                        continue

                    yield span, match

    def match_with_groupdict_as_spans(
        self,
        doclike: Union[Doc, Span],
    ) -> Tuple[Span, Dict[str, Span]]:
        """
        Iterates on the matches.

        Parameters
        ----------
        doclike:
            spaCy Doc or Span object to match on.

        Yields
        -------
        span:
            A match.
        """

        for key, patterns, attr, ignore_excluded, alignment_mode in self.regex:
            text = get_text(doclike, attr, ignore_excluded)

            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.trace(f"Matched a regex from {key}: {repr(match.group())}")

                    start_char, end_char = span_from_match(
                        match=match,
                        span_from_group=self.span_from_group,
                    )

                    span = create_span(
                        doclike=doclike,
                        start_char=start_char,
                        end_char=end_char,
                        key=key,
                        attr=attr,
                        alignment_mode=alignment_mode,
                        ignore_excluded=ignore_excluded,
                    )
                    group_spans = {}
                    for group_key, group_string in match.groupdict().items():
                        if group_string:
                            group_spans[group_key] = create_span(
                                doclike=doclike,
                                start_char=match.start(group_key),
                                end_char=match.end(group_key),
                                key=group_key,
                                attr=attr,
                                alignment_mode=alignment_mode,
                                ignore_excluded=ignore_excluded,
                            )

                    yield span, group_spans

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
            spaCy Doc or Span object.
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
