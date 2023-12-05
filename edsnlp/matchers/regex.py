import re
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from spacy.tokens import Doc, Span

from edsnlp.utils.doc_to_text import get_char_offsets, get_text
from edsnlp.utils.regex import compile_regex

from .utils import Patterns


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
    ignore_space_tokens: bool,
) -> Optional[Span]:
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
    ignore_space_tokens : bool
        Whether to skip space tokens.

    Returns
    -------
    span:
        A span matched on the Doc-like object.
    """

    doc = doclike if isinstance(doclike, Doc) else doclike.doc

    # Handle the simple case immediately
    if attr in {"TEXT", "LOWER"} and not ignore_excluded and not ignore_space_tokens:
        off = doclike[0].idx
        return doc.char_span(
            start_char + off,
            end_char + off,
            label=key,
            alignment_mode=alignment_mode,
        )

    clean_starts, clean_ends = get_char_offsets(
        doc,
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
    )

    first = clean_starts[doclike[0].i]

    start_char += first
    end_char += first

    # (rightmost token that starts right after the start_char, but not at it)
    start_i = bisect_right(clean_starts, start_char)
    # (leftmost token that ends before or at the end_char) + 1
    end_i = bisect_left(clean_ends, end_char)

    if alignment_mode == "expand":
        # if the start_char is inside the previous token (<end), we want to include it
        if start_i > 0 and start_char < clean_ends[start_i - 1]:
            start_i -= 1
        # if the end_char is inside the next token (>start), we want to include it
        if end_i < len(clean_starts) and end_char > clean_starts[end_i]:
            end_i += 1
    elif alignment_mode == "strict":
        start_i -= 1
        if (
            start_i >= 0
            and start_char != clean_starts[start_i]
            or end_char != clean_ends[end_i]
        ):
            return None
        end_i += 1
    elif alignment_mode == "contract":
        # if start_char is before the previous token (<=start), we want to include it
        if start_i > 0 and start_char <= clean_starts[start_i - 1]:
            start_i -= 1
        # if end_char is after the next token (>=end), we want to include it
        if end_i < len(clean_ends) and clean_ends[end_i] <= end_char:
            end_i += 1

        # Compatibility with spacy behavior, but ideally we would return an empty span
        if end_i <= start_i:
            return None
    else:
        raise ValueError(f"Invalid alignment mode: {alignment_mode}")

    return Span(doc, start_i, max(start_i, end_i), label=key)


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
    ignore_space_tokens: bool
        Whether to skip space tokens during matching.

        You won't be able to match on newlines if this is enabled and
        the "spaces"/"newline" option of `eds.normalizer` is enabled (by default).
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
        ignore_space_tokens: bool = False,
        flags: Union[re.RegexFlag, int] = 0,  # No additional flags
        span_from_group: bool = False,
    ):
        self.alignment_mode = alignment_mode
        self.regex = []

        self.default_attr = attr

        self.flags = flags
        self.span_from_group = span_from_group

        self.ignore_excluded = ignore_excluded
        self.ignore_space_tokens = ignore_space_tokens

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
        ignore_space_tokens: Optional[bool] = None,
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
        attr : Optional[str]
            Attribute to use for matching.
            By default, uses the `default_attr` attribute
        ignore_excluded : Optional[bool]
            Whether to skip excluded tokens during matching.
        ignore_space_tokens: Optional[bool]
            Whether to skip space tokens during matching.

            You won't be able to match on newlines if this is enabled and
            the "spaces"/"newline" option of `eds.normalizer` is enabled (by default).

        alignment_mode : Optional[str]
            Overwrite alignment mode.
        """

        if attr is None:
            attr = self.default_attr

        if ignore_excluded is None:
            ignore_excluded = self.ignore_excluded

        if ignore_space_tokens is None:
            ignore_space_tokens = self.ignore_space_tokens

        if alignment_mode is None:
            alignment_mode = self.alignment_mode

        if flags is None:
            flags = self.flags

        patterns = [compile_regex(pattern, flags) for pattern in patterns]

        self.regex.append(
            (
                key,
                patterns,
                attr,
                ignore_excluded,
                ignore_space_tokens,
                alignment_mode,
            )
        )

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
        self.regex = [pat for pat in self.regex if pat[0] != key]
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

        for (
            key,
            patterns,
            attr,
            ignore_excluded,
            ignore_space_tokens,
            alignment_mode,
        ) in self.regex:
            text = get_text(doclike, attr, ignore_excluded, ignore_space_tokens)

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
                        ignore_space_tokens=ignore_space_tokens,
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

        for (
            key,
            patterns,
            attr,
            ignore_excluded,
            ignore_space_tokens,
            alignment_mode,
        ) in self.regex:
            text = get_text(doclike, attr, ignore_excluded, ignore_space_tokens)

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
                        ignore_space_tokens=ignore_space_tokens,
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
                                ignore_space_tokens=ignore_space_tokens,
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
