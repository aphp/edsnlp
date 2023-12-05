from functools import lru_cache
from typing import List, Tuple, Union

import spacy.attrs
from spacy.tokens import Doc, Span


@lru_cache(32)
def aggregate_tokens(
    doc: Doc,
    attr: str,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
) -> Tuple[str, List[int], List[int], bytes]:
    """
    Aggregate tokens strings, computed from their `attr` attribute, into a single
    string, possibly ignoring excluded tokens (like pollution tokens) and/or space
    tokens. This also returns the start and end offsets of each token in the
    aggregated string, as well as a bytes array indicating which tokens were kept.
    The reason for the bytes array is that it is faster to index, and allows reverse
    indexing as well.

    Parameters
    ----------
    doc: Doc
    attr: str
    ignore_excluded: bool
    ignore_space_tokens: bool

    Returns
    -------
    Tuple[str, List[int], List[int], bytes]
        The aggregated text, the start offsets, the end offsets, and the bytes array
        indicating which tokens were kept.
    """
    idx_to_strings = doc.vocab.strings
    exclude_hash = idx_to_strings["EXCLUDED"]
    space_hash = idx_to_strings["SPACE"]
    spacy_attr = "ORTH" if attr in {"TEXT", "LOWER"} else attr
    offset = 0
    if not ignore_excluded and not ignore_space_tokens and spacy_attr == "ORTH":
        arr = doc.to_array(
            [spacy.attrs.SPACY, spacy.attrs.ORTH, spacy.attrs.IDX, spacy.attrs.LENGTH]
        )
        text_parts = [""] * len(arr)
        for i, (str_hash, space) in enumerate(
            zip(arr[:, 1].tolist(), arr[:, 0].tolist())
        ):
            text_parts[i] = idx_to_strings[str_hash] + (" " if space else "")
        begins = arr[:, 2].tolist()
        ends = (arr[:, 2] + arr[:, 3]).tolist()
        keep_list = [True] * len(arr)
    else:
        if hasattr(spacy.attrs, spacy_attr):
            arr = doc.to_array(
                [
                    spacy.attrs.SPACY,
                    spacy.attrs.TAG,
                    getattr(spacy.attrs, spacy_attr),
                ]
            )
            tokens_space = arr[:, 0].tolist()
            tokens_tag = arr[:, 1]
            tokens_text = arr[:, 2].tolist()
        # else:
        #     arr = doc.to_array([spacy.attrs.SPACY, spacy.attrs.TAG])
        #     tokens_space = arr[:, 0].tolist()
        #     tokens_tag = arr[:, 1]
        #     tokens_text = [token._.get(spacy_attr) for token in doc]

        text_parts = [""] * len(arr)
        begins = [0] * len(arr)
        ends = [0] * len(arr)
        if ignore_excluded and ignore_space_tokens:
            keep_list = (
                (tokens_tag != exclude_hash) & (tokens_tag != space_hash)
            ).tolist()
        elif ignore_excluded:
            keep_list = (tokens_tag != exclude_hash).tolist()
        elif ignore_space_tokens:
            keep_list = (tokens_tag != space_hash).tolist()
        else:
            keep_list = [True] * len(arr)

        for i, (str_hash, space, keep) in enumerate(
            zip(tokens_text, tokens_space, keep_list)
        ):
            if keep:
                if space:
                    part = idx_to_strings[str_hash] + " "
                    text_parts[i] = part
                    begins[i] = offset
                    offset += len(part)
                    ends[i] = offset - 1
                else:
                    part = idx_to_strings[str_hash]
                    text_parts[i] = part
                    begins[i] = offset
                    offset += len(part)
                    ends[i] = offset
            else:
                begins[i] = offset
                ends[i] = offset

    text = "".join(text_parts)
    if attr == "LOWER":
        text = text.lower()
    return text, begins, ends, bytes(keep_list)


def get_text(
    doclike: Union[Doc, Span],
    attr: str,
    ignore_excluded: bool,
    ignore_space_tokens: bool = False,
) -> str:
    """
    Get text using a custom attribute, possibly ignoring excluded tokens.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        Doc or Span to get text from.
    attr : str
        Attribute to use.
    ignore_excluded : bool
        Whether to skip excluded tokens, by default False
    ignore_space_tokens : bool
        Whether to skip space tokens, by default False

    Returns
    -------
    str
        Extracted text.
    """
    is_doc = isinstance(doclike, Doc)
    text, starts, ends, keep = aggregate_tokens(
        doclike if is_doc else doclike.doc,
        attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
    )
    try:
        return (
            text[
                starts[keep.index(1, doclike[0].i)] : ends[
                    keep.rindex(1, None, doclike[-1].i + 1)
                ]
            ]
            if len(doclike)
            else ""
        )
    except ValueError:
        return ""


def get_char_offsets(
    doclike: Union[Doc, Span],
    attr: str,
    ignore_excluded: bool,
    ignore_space_tokens: bool = False,
) -> tuple:
    """
    Get char offsets of the doc tokens in the "cleaned" text.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        Doc or Span to get text from.
    attr : str
        Attribute to use.
    ignore_excluded : bool
        Whether to skip excluded tokens, by default False
    ignore_space_tokens : bool
        Whether to skip space tokens, by default False

    Returns
    -------
    Tuple[List[int], List[int]]
        An alignment tuple: clean start/end offsets lists.
    """
    return aggregate_tokens(
        doclike if isinstance(doclike, Doc) else doclike.doc,
        attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
    )[1:3]
