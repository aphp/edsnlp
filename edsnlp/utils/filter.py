from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

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
    spans: Iterable[Union["Span", Tuple["Span", Any]]],
    label_to_remove: Optional[str] = None,
    return_discarded: bool = False,
) -> Union[List["Span"], Tuple[List["Span"], List["Span"]]]:
    """
    Re-definition of spacy's filtering function, that returns discarded spans
    as well as filtered ones.

    Can also accept a `label_to_remove` argument, useful for filtering out
    pseudo cues. If set, `results` can contain overlapping spans: only
    spans overlapping with excluded labels are removed. The main expected
    use case is for pseudo-cues.

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
    spans : List[Span]
        Spans to filter.
    return_discarded : bool
        Whether to return discarded spans.
    label_to_remove : str, optional
        Label to remove. If set, results can contain overlapping spans.

    Returns
    -------
    results : List[Span]
        Filtered spans
    discarded : List[Span], optional
        Discarded spans
    """
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    discarded = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            if label_to_remove is None or span.label_ != label_to_remove:
                result.append(span)
            if label_to_remove is None or span.label_ == label_to_remove:
                seen_tokens.update(range(span.start, span.end))
        elif label_to_remove is None or span.label_ != label_to_remove:
            discarded.append(span)

    result = sorted(result, key=lambda span: span.start)
    discarded = sorted(discarded, key=lambda span: span.start)

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
