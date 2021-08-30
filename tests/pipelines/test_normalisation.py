from edsnlp.pipelines.normalizer.normalizer import (
    Normalizer,
    replace,
    accents,
    quotes_and_apostrophes,
)

import edsnlp.components

from pytest import fixture


@fixture
def doc(nlp):
    text = "Le patient ʺnˊest pas malade”, écrit-il."
    return nlp(text)


def test_replace():
    text = "üîéè"
    assert replace(text, accents) == "uiee"


def test_normalization_quotes_and_apostrophes(doc):

    normalizer = Normalizer(
        lowercase=False,
        remove_accents=False,
        normalize_quotes=True,
    )

    doc = normalizer(doc)
    norm = "".join([t.norm_ + t.whitespace_ for t in doc])

    assert norm == 'Le patient "n\'est pas malade", écrit-il.'


def test_normalization_accents(doc):

    normalizer = Normalizer(
        lowercase=False,
        remove_accents=True,
        normalize_quotes=True,
    )

    doc = normalizer(doc)
    norm = "".join([t.norm_ + t.whitespace_ for t in doc])

    assert norm == 'Le patient "n\'est pas malade", ecrit-il.'


def test_normalization_lowercase(doc):

    normalizer = Normalizer(
        lowercase=True,
        remove_accents=False,
        normalize_quotes=True,
    )

    doc = normalizer(doc)
    norm = "".join([t.norm_ + t.whitespace_ for t in doc])

    assert norm == 'le patient "n\'est pas malade", écrit-il.'


def test_pipeline_component(nlp, doc):
    normalizer = nlp.add_pipe("normalizer")
    doc = normalizer(doc)

    norm = "".join([t.norm_ + t.whitespace_ for t in doc])
    assert norm == 'le patient "n\'est pas malade", ecrit-il.'
