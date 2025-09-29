from typing import Union

import numpy as np
import pytest
from spacy.tokens import Doc

import edsnlp
from edsnlp.utils.fuzzy_alignment import AnnotatedText, align, regex_multisub_with_spans


def test_regex_multi_sub_with_spans():
    kw = dict(
        patterns=["===", r"hello (world)", "example"],
        replacements=[" ", r"hi \1", "sample"],
        text="=== hello world! This is a hello world example.===",
    )
    new_text, deltas = regex_multisub_with_spans(**kw)
    assert new_text == "  hi world! This is a hi world sample. "
    new_text, deltas = regex_multisub_with_spans(**kw, return_deltas=True)
    assert new_text == "  hi world! This is a hi world sample. "
    # entities=[
    #     {"label": "GREET", "fragments": [{"begin": 4, "end": 9}]},
    #     {"label": "WORLD", "fragments": [{"begin": 10, "end": 16}]},
    #     {"label": "EXAMPLE", "fragments": [{"begin": 39, "end": 46}]},
    # ]
    new_begins = deltas.apply(np.asarray([4, 10, 39]), side="left")
    new_ends = deltas.apply(np.asarray([9, 16, 46]), side="right")
    assert (new_begins == np.asarray([2, 2, 31])).all()
    # since we replace "hello world" by "hi world", the
    # span "hello" becomes "hi world", hence 10 instead of 4
    assert (new_ends == np.asarray([10, 11, 37])).all()

    old_begins = deltas.unapply(np.asarray([2, 5, 31, 33]), side="left")
    old_ends = deltas.unapply(np.asarray([10, 11, 37, 35]), side="right")
    assert (old_begins == np.asarray([4, 4, 39, 39])).all()
    assert (old_ends == np.asarray([15, 16, 46, 46])).all()


def conv_from_doc(doc: Union[Doc, str]):
    if isinstance(doc, str):
        return {"doc_id": None, "text": doc, "entities": []}
    return {
        "doc_id": doc._.note_id,
        "text": doc.text,
        "entities": [
            {
                "label": ent.label_,
                "fragments": [{"begin": ent.start_char, "end": ent.end_char}],
            }
            for ent in doc.ents
        ],
    }


def conv_to_doc(annotated: AnnotatedText, tok=None) -> Doc:
    if tok is None:
        tok = edsnlp.data.converters.get_current_tokenizer()
    doc = tok(annotated["text"])
    ents = []
    for ent in annotated["entities"]:
        for frag in ent["fragments"]:
            span = doc.char_span(
                frag["begin"], frag["end"], label=ent["label"], alignment_mode="expand"
            )
            if span is not None:
                ents.append(span)
    doc.ents = ents
    doc._.note_id = annotated["doc_id"]
    return doc


@pytest.mark.parametrize("high_threshold", [True, False])
def test_align(high_threshold, md2doc, doc2md):
    old = (
        "This is a [small sample](SAMPLE)\n\n[\ndoc\n](DOC)  . "
        "It contains some [texts](TEXT) to be annotated. "
    )
    new = (
        "This is a  modified  [small   sample](SAMPLE) [\ndoc\n](DOC) . "
        "It contains some [text](TEXT) to be annotated."
    )
    if not high_threshold:
        res = align(
            conv_from_doc(md2doc(old)),
            conv_from_doc(md2doc(new)),
            do_debug=True,
        )
        result = conv_to_doc(res["doc"])
        assert doc2md(result) == new
    else:
        res = align(
            conv_from_doc(md2doc(old)),
            conv_from_doc(md2doc(new)),
            do_debug=True,
            threshold=20,
        )
        result = conv_to_doc(res["doc"])
        assert doc2md(result) == md2doc(new).text


def test_ambiguous_exact_match(md2doc, doc2md):
    old = (
        "this is ambiguous: xy , where should we annotate ?\n" * 25
        + "this is ambiguous [xy](AMBIGUOUS) , where should we annotate ?\n"
        + "this is ambiguous: xy , where should we annotate ?\n" * 24
    )
    new = "this is ambiguous: xy , where should we annotate ?\n" * 50
    res = align(
        conv_from_doc(md2doc(old)),
        conv_from_doc(md2doc(new)),
        do_debug=True,
        sim_scheme=[(10, 0.7)],
    )
    result = conv_to_doc(res["doc"])
    assert doc2md(result) == new


def test_ambiguous_inexact_match(md2doc, doc2md):
    old = (
        "this is ambiguous: xy , where should we annotate ?\n" * 25
        + "this is ambiguous [xz](AMBIGUOUS) , where should we annotate ?\n"
        + "this is ambiguous: xy , where should we annotate ?\n" * 24
    )
    new = "this is ambiguous: xy , where should we annotate ?\n" * 50
    res = align(
        conv_from_doc(md2doc(old)),
        conv_from_doc(md2doc(new)),
        do_debug=True,
        sim_scheme=[(10, 0.7)],
    )
    result = conv_to_doc(res["doc"])
    assert doc2md(result) == new


def test_missing(md2doc, doc2md):
    old = "this is a sample: [abcd](MISSING), where[](EMPTY) should we annotate ?"
    new = "the cat sat on the mat, and eat mice when he is hungry."
    res = align(
        conv_from_doc(md2doc(old)),
        conv_from_doc(md2doc(new)),
        do_debug=True,
        sim_scheme=[(10, 0.7)],
    )
    result = conv_to_doc(res["doc"])
    assert doc2md(result) == new
