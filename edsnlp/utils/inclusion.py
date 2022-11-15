from spacy.tokens import Span


def check_inclusion(span: Span, start: int, end: int) -> bool:
    """
    Checks whether the span overlaps the boundaries.

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
        Whether the span overlaps the boundaries.
    """

    if span.start >= end or span.end <= start:
        return False
    return True


def check_sent_inclusion(span: Span, start: int, end: int) -> bool:
    """
    Checks whether the span overlaps the boundaries.

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
        Whether the span overlaps the boundaries.
    """
    if span.sent.start >= end or span.sent.end <= start:
        return False
    return True
