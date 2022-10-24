from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from spacy.tokens import Span


def default_sort_key(span: Span) -> Tuple[int, int]:
    """
    Returns the sort key for filtering spans.

    Parameters
    ----------
    span : Span
        Span to sort.

    Returns
    -------
    key : Tuple(int, int)
        Sort key.
    """
    if isinstance(span, tuple):
        span = span[0]
    return span.end - span.start, -span.start


def start_sort_key(span: Union[Span, Tuple[Span, Any]]) -> Tuple[int, int]:
    """
    Returns the sort key for filtering spans by start order.

    Parameters
    ----------
    span : Span
        Span to sort.

    Returns
    -------
    key : Tuple(int, int)
        Sort key.
    """
    if isinstance(span, tuple):
        span = span[0]
    return span.start


def filter_spans(
    spans: Iterable[Union["Span", Tuple["Span", Any]]],
    label_to_remove: Optional[str] = None,
    return_discarded: bool = False,
    sort_key: Callable[[Span], Any] = default_sort_key,
) -> Union[
    List[Union[Span, Tuple[Span, Any]]],
    Tuple[List[Union[Span, Tuple[Span, Any]]], List[Union[Span, Tuple[Span, Any]]]],
]:
    """
    Re-definition of spacy's filtering function, that returns discarded spans
    as well as filtered ones.

    Can also accept a `label_to_remove` argument, useful for filtering out
    pseudo cues. If set, `results` can contain overlapping spans: only
    spans overlapping with excluded labels are removed. The main expected
    use case is for pseudo-cues.

    It can handle an iterable of tuples instead of an iterable of `Span`s.
    The primary use-case is the use with the `RegexMatcher`'s capacity to
    return the span's `groupdict`.

    !!! note ""

        The **spaCy documentation states**:

        > Filter a sequence of spans and remove duplicates or overlaps.
        > Useful for creating named entities (where one token can only
        > be part of one entity) or when merging spans with
        > `Retokenizer.merge`. When spans overlap, the (first)
        > longest span is preferred over shorter spans.

    !!! danger "Filtering out spans"

        If the `label_to_remove` argument is supplied, it might be tempting to
        filter overlapping spans that are not part of a label to remove.

        The reason we keep all other possibly overlapping labels is that in qualifier
        pipelines, the same cue can precede **and** follow a marked entity.
        Hence we need to keep every example.

    Parameters
    ----------
    spans : Iterable[Union["Span", Tuple["Span", Any]]]
        Spans to filter.
    return_discarded : bool
        Whether to return discarded spans.
    label_to_remove : str, optional
        Label to remove. If set, results can contain overlapping spans.
    sort_key : Callable[Span, Any], optional
        Key to sorting spans before applying overlap conflict resolution.
        A span with a higher key will have precedence over another span.
        By default, the largest, leftmost spans are selected first.

    Returns
    -------
    results : List[Union[Span, Tuple[Span, Any]]]
        Filtered spans
    discarded : List[Union[Span, Tuple[Span, Any]]], optional
        Discarded spans
    """
    sorted_spans = sorted(spans, key=sort_key, reverse=True)
    result = []
    discarded = []
    seen_tokens = set()
    for span in sorted_spans:
        s = span if isinstance(span, Span) else span[0]
        # Check for end - 1 here because boundaries are inclusive
        token_range = set(range(s.start, s.end))
        if token_range.isdisjoint(seen_tokens):
            if label_to_remove is None or s.label_ != label_to_remove:
                result.append(span)
            if label_to_remove is None or s.label_ == label_to_remove:
                seen_tokens.update(token_range)
        elif label_to_remove is None or s.label_ != label_to_remove:
            discarded.append(span)

    result = sorted(result, key=start_sort_key)
    discarded = sorted(discarded, key=start_sort_key)

    if return_discarded:
        return result, discarded

    return result


def consume_spans(
    spans: List[Span],
    filter: Callable,
    second_chance: Optional[List[Span]] = None,
) -> Tuple[List[Span], List[Span]]:
    """
    Consume a list of span, according to a filter.

    !!! warning
        This method makes the hard hypothesis that:

        1. Spans are sorted.
        2. Spans are consumed in sequence and only once.

        The second item is problematic for the way we treat long entities,
        hence the `second_chance` parameter, which lets entities be seen
        more than once.

    Parameters
    ----------
    spans : List of spans
        List of spans to filter
    filter : Callable
        Filtering function. Should return True when the item is to be included.
    second_chance : List of spans, optional
        Optional list of spans to include again (useful for long entities),
        by default None

    Returns
    -------
    matches : List of spans
        List of spans consumed by the filter.
    remainder : List of spans
        List of remaining spans in the original `spans` parameter.
    """

    if not second_chance:
        second_chance = []
    else:
        second_chance = [m for m in second_chance if filter(m)]

    if not spans:
        return second_chance, []

    for i, span in enumerate(spans):
        if not filter(span):
            break
        else:
            i += 1

    matches = spans[:i]
    remainder = spans[i:]

    matches.extend(second_chance)

    return matches, remainder


def get_spans(spans: List[Span], label: Union[int, str]) -> List[Span]:
    """
    Extracts spans with a given label.
    Prefer using hash label for performance reasons.

    Parameters
    ----------
    spans : List[Span]
        List of spans to filter.
    label : Union[int, str]
        Label to filter on.

    Returns
    -------
    List[Span]
        Filtered spans.
    """
    if isinstance(label, int):
        return [span for span in spans if span.label == label]
    else:
        return [span for span in spans if span.label_ == label]
