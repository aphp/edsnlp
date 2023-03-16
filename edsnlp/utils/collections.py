def dedup(sequence, key=None):
    """
    Deduplicate a sequence, keeping the last occurrence of each item.

    Parameters
    ----------
    sequence : Sequence
        Sequence to deduplicate
    key : Callable, optional
        Key function to use for deduplication, by default None
    """
    key = (lambda x: x) if key is None else key
    return list({key(item): item for item in sequence}.values())
