from typing import List, Tuple

from spacy.tokens import Span, Token

from edsnlp.matchers.utils import ATTRIBUTES
from edsnlp.utils.extensions import rgetattr


def to_attr(
    tokens: List[Tuple[Token, str]],
    attr: str = "TEXT",
) -> List[Tuple[Token, str]]:
    """
    Converts a list of (token, whitespace) tuples to string

    Parameters
    ----------
    tokens : List[Tuple[Token, str]]
        The list of (token, whitespace) tuples to string
    attr : str, optional
        The attribute to use, by default "TEXT"

    Returns
    -------
    List[Tuple[Token, str]]
        The list with adapted tokens
    """

    attr = attr.upper()
    attr = ATTRIBUTES.get(attr, attr)

    return [(rgetattr(token, attr, ""), whitespace) for (token, whitespace) in tokens]


def to_text(
    tokens: List[Tuple[Token, str]],
) -> str:
    """
    Converts a list of (token, whitespace) tuples to string

    Parameters
    ----------
    tokens : List[Tuple[Token, str]]
        The list of (token, whitespace) tuples to string

    Returns
    -------
    str
        The string obtained from the input list
    """
    return "".join(token + whitespace for (token, whitespace) in tokens)


def get_ent_sentences(
    ent: Span,
    n_before: int,
    n_after: int,
) -> Span:
    """
    Extract the span composed of
    `n_before` sentences before `ent`, and `n_after` after `ent`

    Parameters
    ----------
    ent : Span
        An entity
    n_before : int
        Number of sentences to extract before
    n_after : int
        Number of sentences to extract after

    Returns
    -------
    Span
       The extracted span
    """
    after = get_ent_next_sentences(ent, n=n_after)
    before = get_ent_previous_sentences(ent, n=n_before)

    return ent.doc[before.start : after.end]


def get_ent_next_sentences(
    ent: Span,
    n: int = 0,
) -> Span:
    if n == 0:
        return ent.doc[ent[0].sent.start : ent[-1].sent.end]

    doc = ent.doc
    sent = ent[0].sent  # sentence to the left

    has_following = ent[-1].sent[-1].i < len(doc) - 1

    if not has_following:
        return ent.doc[ent[0].sent.start : ent[-1].sent.end]  # ent

    following = ent[-1].sent[-1].nbor(1).sent
    following_end = following.end
    return get_ent_next_sentences(doc[sent.start : following_end], n=n - 1)


def get_ent_previous_sentences(
    ent: Span,
    n: int = 0,
) -> Span:
    if n == 0:
        return ent.doc[ent[0].sent.start : ent[-1].sent.end]

    doc = ent.doc
    sent = ent[-1].sent  # sentence to the right

    has_previous = ent[0].sent[0].i > 0

    if not has_previous:
        return ent.doc[ent[0].sent.start : ent[-1].sent.end]  # ent

    previous = ent[0].sent[0].nbor(-1).sent
    previous_start = previous.start

    return get_ent_previous_sentences(doc[previous_start : sent.end], n=n - 1)


def get_span_unitary(
    ent: Span,
    n: int = 10,
    reverse: bool = False,
    ignore_excluded: bool = True,
    count_excluded: bool = False,
) -> List[Tuple[Token, str]]:
    """
    Get a list of k tokens after or before an entity.

    Parameters
    ----------
    ent : Span
        the entity / span
    n : int,
        number of tokens to extract, by default 10
    reverse : bool,
        whether to extract tokens before the entity, by default False
    ignore_excluded : bool
        Whether to exclude excluded tokens or not
    count_excluded : bool
        Whether to exclude excluded tokens from the count (parameter n)

    Returns
    -------
    List[Tuple[Token, str]]
        List of (token, whitespace) tuples
    """
    doc = ent.doc
    start = ent.start
    end = ent.end

    words = []

    j = 0

    if n == 0:
        return []
    if reverse:
        iterable = reversed(doc[:start])
    else:
        iterable = doc[end:]

    if ignore_excluded:
        for token in iterable:
            j += (not token._.excluded) or (count_excluded)
            if not token._.excluded:
                words.append((token, token.whitespace_))
            if j == n:
                break
    else:
        for token in iterable:
            j += 1
            words.append((token, token.whitespace_))
            if j == n:
                break
    if reverse:
        words = list(reversed(words))

    return words


def get_span_text_and_offsets(
    ent: Span,
    n_before: int,
    n_after: int,
    return_type: str = "list",
    mode: str = "token",
    attr: str = "TEXT",
    ignore_excluded: bool = True,
):
    """
    Get the surrounding context of an entity.

    Parameters
    ----------
    ent : Span
        the entity / span
    mode : str
        Wheter `n_before` and `n_after` should represent number of sentences
        (`mode="sentence"`) or number of tokens (`mode="token"`), by default "sentence"
    n_before : int,
        Number of tokens / sentences to extract before the entity, by default 1 sentence
    n_after : int,
        Number of tokens / sentences to extract after the entity, by default 1 sentence
    ignore_excluded : bool
        Whether to exclude excluded tokens or not
    attr : str
        Which attribute to use when converting token to string.
        Available: LOWER, TEXT, NORM, SHAPE

    Returns
    -------
    List[Tuple[Token, str]]
        List of (token, whitespace) tuples
    """

    count_excluded = False

    if mode == "sentence":
        sentences = get_ent_sentences(ent, n_before, n_after)
        n_before, n_after = ent.start - sentences.start, sentences.end - ent.end
        count_excluded = True

    preceding = get_span_unitary(
        ent,
        reverse=True,
        n=n_before,
        ignore_excluded=ignore_excluded,
        count_excluded=count_excluded,
    )
    queue = get_span_unitary(
        ent,
        reverse=False,
        n=n_after,
        ignore_excluded=ignore_excluded,
        count_excluded=count_excluded,
    )

    transformed_ent = [(t, t.whitespace_) for t in ent]

    preceding = to_attr(preceding, attr=attr)
    queue = to_attr(queue, attr=attr)
    transformed_ent = to_attr(transformed_ent, attr=attr)

    if return_type == "text":
        ent_trailing_token = transformed_ent[-1]
        transformed_ent[-1] = (ent_trailing_token[0], "")

        queue.insert(0, ("", ent_trailing_token[1]))
        queue[-1] = (queue[-1][0], "")

        preceding = to_text(preceding)
        queue = to_text(queue)
        transformed_ent = to_text(transformed_ent)

    elif return_type == "list":
        preceding = (list(zip(*preceding)) or [(), ()])[0]
        queue = (list(zip(*queue)) or [(), ()])[0]
        transformed_ent = (list(zip(*transformed_ent)) or [(), ()])[0]

    words = preceding + transformed_ent + queue
    span_start = len(preceding)
    span_end = span_start + len(transformed_ent)

    return words, (span_start, span_end)
