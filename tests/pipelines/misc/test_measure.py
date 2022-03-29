import spacy
from pytest import fixture, raises
from spacy.language import Language

from edsnlp.pipelines.misc.measures import Measures
from edsnlp.pipelines.misc.measures.factory import DEFAULT_CONFIG

text = (
    "Le patient fait 1 m 50 kg. La tumeur située à 8h fait 2cm x 3cm. \n"
    "Une autre tumeur plus petite fait 2 par 1mm.\n"
    "Les trois éléments font 8, 13 et 15dm."
)


@fixture
def blank_nlp():
    model = spacy.blank("fr")
    model.add_pipe("eds.sentences")
    return model


@fixture
def measure(blank_nlp: Language):
    return Measures(
        blank_nlp,
        **DEFAULT_CONFIG,
    )


def test_default_factory(blank_nlp: Language):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.measures",
        config=dict(
            measures=["eds.measures.size", "eds.measures.weight", "eds.measures.angle"]
        ),
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["measures"]) == 8


def test_measures_component(blank_nlp: Language, measure: Measures):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["measures"]

    doc = measure(doc)

    m1, m2, m3, m4, m5, m6, m7, m8 = doc.spans["measures"]

    assert str(m1._.value) == "1.0m"
    assert str(m2._.value) == "50.0kg"
    assert str(m3._.value) == "8.0h"
    assert str(m4._.value) == "2.0cm x 3.0cm"
    assert str(m5._.value) == "2.0mm x 1.0mm"
    assert str(m6._.value) == "8.0dm"
    assert str(m7._.value) == "13.0dm"
    assert str(m8._.value) == "15.0dm"


def test_measures_component_scaling(blank_nlp: Language, measure: Measures):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["measures"]

    doc = measure(doc)

    m1, m2, m3, m4, m5, m6, m7, m8 = doc.spans["measures"]

    assert m1._.value.cm == 100.0
    assert m2._.value.mg == 50000000.0
    assert m3._.value.h == 8
    assert m4._.value.mm == (20, 30)
    assert m5._.value.cm == (0.2, 0.1)
    assert m6._.value.dm == 8.0
    assert m7._.value.m == 1.3, (13.0, 100 / 1000, 13.0 * (100 / 1000))
    assert m8._.value.m == 1.5


def test_measure_label(blank_nlp: Language, measure: Measures):
    doc = blank_nlp(text)
    doc = measure(doc)
    m1, m2, m3, m4, m5, m6, m7, m8 = doc.spans["measures"]

    assert m1.label_ == "eds.measures.size"
    assert m2.label_ == "eds.measures.weight"
    assert m3.label_ == "eds.measures.angle"
    assert m4.label_ == "eds.measures.size"
    assert m5.label_ == "eds.measures.size"
    assert m6.label_ == "eds.measures.size"
    assert m7.label_ == "eds.measures.size"
    assert m8.label_ == "eds.measures.size"


def test_measure_str(blank_nlp: Language, measure: Measures):
    for text, res in [
        ("1m50", "1.5m"),
        ("1,50cm", "1.5cm"),
        ("1h45", "1.75h"),
        ("1,45h", "1.45h"),
    ]:
        doc = blank_nlp(text)
        doc = measure(doc)

        assert str(doc.spans["measures"][0]._.value) == res


def test_measure_repr(blank_nlp: Language, measure: Measures):
    for text, res in [
        (
            "1m50",
            "Size(1.5, 'm')",
        ),
        (
            "1,50cm",
            "Size(1.5, 'cm')",
        ),
        (
            "1h45",
            "Angle(1.75, 'h')",
        ),
        (
            "1,45h",
            "Angle(1.45, 'h')",
        ),
        (
            "1 x 2cm",
            "CompositeSize([Size(1.0, 'cm'), Size(2.0, 'cm')])",
        ),
    ]:
        doc = blank_nlp(text)
        doc = measure(doc)

        assert repr(doc.spans["measures"][0]._.value) == res


def test_pooling(blank_nlp: Language, measure: Measures):
    for text, min_res, first_res in [
        ("1m50 x 120cm", "120.0cm", "1.5m"),
        ("150cm30 x 1m x 12dm", "1.0m", "150.3cm"),
        ("1cm50", "1.5cm", "1.5cm"),
    ]:
        doc = blank_nlp(text)
        doc = measure(doc)

        assert str(min(doc.spans["measures"][0]._.value)) == min_res
        assert str(doc.spans["measures"][0]._.value[0]) == first_res


def test_compare(blank_nlp: Language, measure: Measures):
    m1, m2 = "1m50 x 120cm", "150cm"
    m1 = measure(blank_nlp(m1)).spans["measures"][0]
    m2 = measure(blank_nlp(m2)).spans["measures"][0]
    assert max(m1._.value) == max(m2._.value)

    m1, m2 = "1m0", "120cm"
    m1 = measure(blank_nlp(m1)).spans["measures"][0]
    m2 = measure(blank_nlp(m2)).spans["measures"][0]
    assert max(m1._.value) <= max(m2._.value)
    assert max(m2._.value) > max(m1._.value)
