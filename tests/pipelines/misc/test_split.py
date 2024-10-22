from spacy.tokens.span import Span

import edsnlp
import edsnlp.pipes as eds


def test_split_line_jump():
    txt = """This is a test. Another test.

A third test!"""
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.matcher", config={"terms": {"test": "test"}})
    doc = nlp(txt)
    Span.set_extension("test_dict", default={}, force=True)
    doc.ents[0]._.test_dict = {"key": doc.ents[1]}
    doc.ents[1]._.test_dict = {"key": doc.ents[0]}
    doc.ents[2]._.test_dict = {"key": doc.ents[0]}
    subdocs = list(eds.split(regex="\n\n")(doc))
    assert len(subdocs) == 2
    assert subdocs[0].text == "This is a test. Another test.\n\n"
    assert subdocs[1].text == "A third test!"
    assert subdocs[0].ents[0]._.test_dict["key"] == subdocs[0].ents[1]
    assert subdocs[0].ents[1]._.test_dict["key"] == subdocs[0].ents[0]
    assert subdocs[1].ents[0]._.test_dict == {}


def test_filter():
    txt = """This is a test. Another test."""
    doc = edsnlp.blank("eds")(txt)
    subdocs = list(
        eds.split(
            regex=r"[.!?]\s+()[A-Z]",
            filter_expr='"Another" in doc.text',
        )(doc)
    )
    assert len(subdocs) == 1
    assert subdocs[0].text == "Another test."


def test_max_length():
    txt = (
        "Le patient mange des pates depuis le début du confinement, "
        "il est donc un peu ballonné, mais pense revenir à un régime plus "
        "équilibré en mangeant des légumes et des fruits."
    )
    doc = edsnlp.blank("eds")(txt)
    texts = [d.text for d in eds.split(max_length=12)(doc)]
    assert texts == [
        "Le patient mange des pates depuis le début du confinement, il ",
        "est donc un peu ballonné, mais pense revenir à un régime ",
        "plus équilibré en mangeant des légumes et des fruits.",
    ]
