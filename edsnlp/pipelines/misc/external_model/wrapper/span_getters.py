from functools import partial
from operator import attrgetter
from typing import Any, Dict, Iterable, List, Tuple, Union

import spacy
from spacy.tokens import Doc, Span

from edsnlp.utils.align import get_span_text_and_offsets


@spacy.registry.span_getters("spans-with-context")
def configured_span_wth_context_loader(
    n_before: int,
    n_after: int,
    return_type: str = "list",
    mode: str = "token",
    attr: str = "TEXT",
    ignore_excluded: bool = True,
    with_ents: bool = True,
    with_spangroups: Union[bool, Iterable[str]] = True,
    output_keys: Dict[str, str] = {"text": "text", "span": "span"},
) -> Tuple[Dict[str, Any], Span]:

    return partial(
        span_wth_context_loader,
        n_before=n_before,
        n_after=n_after,
        return_type=return_type,
        mode=mode,
        attr=attr,
        ignore_excluded=ignore_excluded,
        with_ents=with_ents,
        with_spangroups=with_spangroups,
        output_keys=output_keys,
    )


def span_wth_context_loader(
    docs: Iterable[Doc],
    mode: str = "sentence",
    n_before: int = 1,
    n_after: int = 1,
    return_type: str = "text",
    attr: str = "TEXT",
    ignore_excluded: bool = True,
    with_ents: bool = True,
    with_spangroups: Union[bool, Iterable[str]] = True,
    output_keys: Dict[str, str] = {"text": "text", "span": "span"},
) -> Tuple[Dict[str, Any], List[Span]]:
    """
    For each entity of each document, return the surrounding context.

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
    return_type: str,
        Wether to return test (`text`) or a list of tokens (`list`), by default `text`
    with_ents: bool,
        Wether to include entities from `doc.ents`, by default True
    with_spangroups: Union[bool, Iterable[str]]
        Wether to include entities stored in `doc.spans`. Use `True` to use all `keys`,
        or a list of keys.
    ignore_excluded : bool
        Whether to exclude excluded tokens or not
    attr : str
        Which attribute to use when converting token to string.
        Available: LOWER, TEXT, NORM, SHAPE
    output_key: Dict[str, str]
        Mapping for the output label. By default, output keys are "text" and "span"

    Returns
    -------
    Tuple[Dict[str, Any], List[Span]]
        Tuple of (output, corresponding_span)
    """
    ents = []
    texts = []

    configured_get_span_text_and_offsets = partial(
        get_span_text_and_offsets,
        n_before=n_before,
        n_after=n_after,
        return_type=return_type,
        mode=mode,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )

    for doc in docs:
        ents.extend(
            get_spans(
                doc,
                with_ents,
                with_spangroups,
            )
        )

    if not ents:
        output = {
            output_keys.get("text", "text"): [],
            output_keys.get("span", "span"): [],
        }
    else:
        texts, spans = zip(*map(configured_get_span_text_and_offsets, ents))

        output = {
            output_keys.get("text", "text"): list(texts),
            output_keys.get("span", "span"): list(spans),
        }
    return (output, ents)


@spacy.registry.span_getters("sentences")  # noqa: F811
def configured_sentences_loader():
    return sentences_loader


def sentences_loader(docs: Iterable[Doc]):
    sents_txt, sents = zip(*((s.text, s) for doc in docs for s in doc.sents))
    return (dict(text=sents_txt), sents)


@spacy.registry.span_getters("docs")  # noqa: F811
def configured_docs_loader():
    return docs_loader


def docs_loader(docs: Iterable[Doc]):
    text = []
    text = [doc.text for doc in docs]
    return (dict(text=text), docs)


def get_spans(
    doc: Doc, with_ents: bool = True, with_spangroups: Union[bool, Iterable[str]] = True
):
    """
    Returns sorted spans of interest depending on the eventual `on_ents_only` value.
    Includes `doc.ents` by default, and adds eventual SpanGroups.
    """
    ents = list(doc.ents) + list(doc.spans.get("discarded", [])) if with_ents else []

    if not with_spangroups:
        return sorted(list(set(ents)), key=(attrgetter("start", "end")))
    if with_spangroups is True:
        with_spangroups = list(doc.spans.keys())
    with_spangroups = set(with_spangroups)

    for spankey in with_spangroups:
        ents.extend(doc.spans.get(spankey, []))

    return sorted(list(set(ents)), key=(attrgetter("start", "end")))
