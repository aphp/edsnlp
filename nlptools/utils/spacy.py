from spacy.tokens import Span

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

    return ((span1.start>=span2.start) & (span1.end<=span2.end))