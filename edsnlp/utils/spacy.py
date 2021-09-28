from spacy.tokens import Span


def check_inclusion(span: Span, start: int, end: int) -> bool:
    """
    Checks whether the span is included in the boundaries.

    Parameters
    ----------
    span : Span
        Span to check.
    start : int
        Start of the boundary
    end : int
        End of the boundary

    Returns
    -------
    bool
        Whether the span is included.
    """

    if span.start > end or span.end < start:
        return False
    return True


def check_spans_inclusion(span1: Span, span2: Span) -> bool:
    """
    Check if a Span in included in another Span

    Parameters
    ----------
    span1:
        Span
    span2:
        Span

    Returns
    -------
    inclusion_check:
        Boolean set to True if span1 in span2, false else
    """

    return (span1.start >= span2.start) & (span1.end <= span2.end)


def span_from_span(
    span: Span, start_idx: int, end_idx: int, label: str, alignment_mode: str = "expand"
) -> Span:
    """
    Create a `Span` object from the slice `span.text[start : end]`.

    Parameters
    ----------
    start (int):
        The index of the first character of the span.
    end (int):
        The index of the first character after the span.
    label (str):
        The label to add to the created Span
    alignment_mode (str) :
        See the doc for `doc.char_span`

    Returns
    -------
        (Span): The newly constructed object.
    """

    start_idx += span.start_char
    end_idx += span.start_char

    return span.doc.char_span(
        start_idx, end_idx, label=label, alignment_mode=alignment_mode
    )
