import copy
import pickle

import dill
import pytest
from spacy.tokens import Doc

import edsnlp

if not hasattr(Doc, "custom_test_reducers"):
    Doc.set_extension("custom_test_reducers", default=None)


@pytest.mark.parametrize("pickler_module", ["pickle", "dill"])
def test_reducers(pickler_module):
    pickler = pickle if pickler_module == "pickle" else dill
    doc = edsnlp.blank("eds")("This is a test.")
    doc.spans["test"] = [doc[0:1]]
    span = doc.spans["test"][0]
    token = doc[0]
    data = [
        doc,  # data[0]
        token,  # data[1]
        token,  # data[2]
        span,  # data[3]
        span,  # data[4]
    ]
    doc._.custom_test_reducers = data
    doc_bytes = pickler.dumps(doc)

    doc_bis = pickler.loads(doc_bytes)
    data_bis = doc_bis._.custom_test_reducers
    assert data_bis[0] is doc_bis
    assert data_bis[1] == doc_bis[0]
    assert data_bis[1] is data_bis[2]
    assert data_bis[3] is data_bis[4]

    doc_ter = pickler.loads(doc_bytes)
    data_ter = doc_ter._.custom_test_reducers
    assert data_ter[0] is not data_bis[0]


def test_deep_copy():
    doc = edsnlp.blank("eds")("This is a test.")
    doc.spans["test"] = [doc[0:1]]
    span = doc.spans["test"][0]
    token = doc[0]
    data = [
        doc,  # data[0]
        token,  # data[1]
        token,  # data[2]
        span,  # data[3]
        span,  # data[4]
    ]

    bis = copy.deepcopy(data)
    ter = copy.deepcopy(data)
    assert bis[1].doc is bis[0]
    assert bis[3].doc is bis[0]
    assert bis[1] is bis[2]
    assert bis[3] is bis[4]

    assert bis[0] is not doc
    assert bis[0] is not ter[0]
