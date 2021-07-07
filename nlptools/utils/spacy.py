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

    return (span1.start>=span2.start) & (span1.end<=span2.end)
    
def span_from_span(span: Span, start_idx: int, alignment_mode: str='expand') -> Span:
    """
    Create a `Span` object from the slice `span.text[start : end]`.
    
    Parameters
    ----------
    start (int): 
        The index of the first character of the span.
    end (int): 
        The index of the first character after the span.
    alignment_mode (str) : 
        See the doc for `doc.char_span`
    
    Returns
    -------
        (Span): The newly constructed object.
    """
    
    start_idx += span.start_char
    end_idx += span.start_char

    return span.doc.char_span(start_idx, end_idx, label=label, kb_id=kb_id, vector=vector, alignment_mode=alignment_mode)