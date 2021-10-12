from typing import Iterable, List, Tuple

from spacy.tokens import Span


def get_sort_key(span: Span) -> Tuple[int, int]:
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
    return span.end - span.start, -span.start


def filter_spans(
    spans: Iterable["Span"],
    return_discarded: bool = False,
) -> Tuple[List["Span"], List["Span"]]:
    """
    Re-definition of spacy's filtering function, that returns discarded spans
    as well as filtered ones.

    .. note ::

        The **Spacy documentation states**:

            Filter a sequence of spans and remove duplicates or overlaps. Useful for
            creating named entities (where one token can only be part of one entity) or
            when merging spans with ``Retokenizer.merge``. When spans overlap, the (first)
            longest span is preferred over shorter spans.

    Parameters
    ----------
    spans : List[Span]
        Spans to filter.
    return_discarded : bool
        Whether to return discarded spans.

    Returns
    -------
    results : List[Span]
        Filtered spans
    discarded : List[Span]
        Discarded spans
    """
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    discarded = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
        else:
            discarded.append(span)
    result = sorted(result, key=lambda span: span.start)
    discarded = sorted(discarded, key=lambda span: span.start)

    if return_discarded:
        return result, discarded

    return result
