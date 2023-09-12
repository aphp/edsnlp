from itertools import chain

import spacy
from pytest import fixture, raises
from spacy.language import Language
from spacy.tokens.span import Span

from edsnlp.pipelines.misc.measurements import MeasurementsMatcher

text = (
    "Le patient fait 1 m 50 kg. La tumeur fait 2.0cm x 3cm. \n"
    "Une autre tumeur plus petite fait 2 par 1mm.\n"
    "Les trois éléments font 8, 13 et 15dm."
)


@fixture
def blank_nlp():
    model = spacy.blank("eds")
    model.add_pipe("eds.normalizer")
    model.add_pipe("eds.sentences")
    return model


@fixture
def matcher(blank_nlp: Language):
    return MeasurementsMatcher(
        blank_nlp,
        extract_ranges=True,
    )


def test_default_factory(blank_nlp: Language):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.measurements",
        config=dict(measurements=["size", "weight", "bmi"]),
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["measurements"]) == 9


def test_measurements_component(blank_nlp: Language, matcher: MeasurementsMatcher):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["measurements"]

    doc = matcher(doc)

    m1, m2, m3, m4, m5, m6, m7, m8, m9 = doc.spans["measurements"]

    assert str(m1._.value) == "1 m"
    assert str(m2._.value) == "50 kg"
    assert str(m3._.value) == "2.0 cm"
    assert str(m4._.value) == "3 cm"
    assert str(m5._.value) == "2 mm"
    assert str(m6._.value) == "1 mm"
    assert str(m7._.value) == "8 dm"
    assert str(m8._.value) == "13 dm"
    assert str(m9._.value) == "15 dm"


def test_measurements_component_scaling(
    blank_nlp: Language, matcher: MeasurementsMatcher
):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["measurements"]

    doc = matcher(doc)

    m1, m2, m3, m4, m5, m6, m7, m8, m9 = doc.spans["measurements"]

    assert m1._.value.cm == 100
    assert m2._.value.mg == 50000000.0
    assert m3._.value.mm == 20
    assert m4._.value.mm == 30
    assert m5._.value.cm == 0.2
    assert m6._.value.cm == 0.1
    assert m7._.value.dm == 8.0
    assert m8._.value.m == 1.3
    assert m9._.value.m == 1.5


def test_measure_label(blank_nlp: Language, matcher: MeasurementsMatcher):
    doc = blank_nlp(text)
    doc = matcher(doc)

    m1, m2, m3, m4, m5, m6, m7, m8, m9 = doc.spans["measurements"]

    assert m1.label_ == "size"
    assert m2.label_ == "weight"
    assert m3.label_ == "size"
    assert m4.label_ == "size"
    assert m5.label_ == "size"
    assert m6.label_ == "size"
    assert m7.label_ == "size"
    assert m8.label_ == "size"
    assert m9.label_ == "size"


def test_measure_str(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res in [
        ("1m50", "1.5 m"),
        ("1,50cm", "1.5 cm"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["measurements"][0]._.value) == res


def test_measure_repr(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res in [
        (
            "1m50",
            "Measurement(1.5, 'm')",
        ),
        (
            "1,50cm",
            "Measurement(1.5, 'cm')",
        ),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        print(doc.spans["measurements"])

        assert repr(doc.spans["measurements"][0]._.value) == res


def test_compare(blank_nlp: Language, matcher: MeasurementsMatcher):
    m1, m2 = "1m0", "120cm"
    m1 = matcher(blank_nlp(m1)).spans["measurements"][0]
    m2 = matcher(blank_nlp(m2)).spans["measurements"][0]
    assert m1._.value <= m2._.value
    assert m2._.value > m1._.value

    m3 = "Entre deux et trois metres"
    m4 = "De 2 à 3 metres"
    m3 = matcher(blank_nlp(m3)).spans["measurements"][0]
    m4 = matcher(blank_nlp(m4)).spans["measurements"][0]
    print(blank_nlp("Entre deux et trois metres"))
    assert str(m3._.value) == "2-3 m"
    assert str(m4._.value) == "2-3 m"
    assert m4._.value.cm == (200.0, 300.0)

    assert m3._.value == m4._.value
    assert m3._.value <= m4._.value
    assert m3._.value >= m1._.value

    assert max(list(chain(m1._.value, m2._.value, m3._.value, m4._.value))).cm == 300


def test_unitless(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res in [
        ("BMI: 24 .", "24 kg_per_m2"),
        ("Le patient mesure 1.5 ", "1.5 m"),
        ("Le patient mesure 152 ", "152 cm"),
        ("Le patient pèse 34 ", "34 kg"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["measurements"][0]._.value) == res


def test_non_matches(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text in [
        "On délivre à 10 g / h.",
        "Le patient grandit de 10 cm par jour ",
        "Truc 10cma truc",
        "01.42.43.56.78 m",
    ]:
        doc = blank_nlp(text)
        print(list(doc))
        doc = matcher(doc)

        assert len(doc.spans["measurements"]) == 0


def test_numbers(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res in [
        ("deux m", "2 m"),
        ("2 m", "2 m"),
        ("⅛ m", "0.125 m"),
        ("0 m", "0 m"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["measurements"][0]._.value) == res


def test_ranges(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res, snippet in [
        ("Le patient fait entre 1 et 2m", "1-2 m", "entre 1 et 2m"),
        ("On mesure de 2 à 2.5 dl d'eau", "2-2.5 dl", "de 2 à 2.5 dl"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        measurement = doc.spans["measurements"][0]
        print(doc.spans["measurements"])
        assert str(measurement._.value) == res
        assert measurement.text == snippet


def test_merge_align(blank_nlp, matcher):
    matcher.merge_mode = "align"
    matcher.span_getter = {"candidates": True}
    matcher.span_setter = {"ents": True}
    doc = blank_nlp(text)
    ent = Span(doc, 10, 15, label="size")
    doc.spans["candidates"] = [ent]
    doc = matcher(doc)

    assert len(doc.ents) == 1
    assert str(ent._.value) == "2.0 cm"


def test_merge_intersect(blank_nlp, matcher: MeasurementsMatcher):
    matcher.merge_mode = "intersect"
    matcher.span_setter = {**matcher.span_setter, "ents": True}
    matcher.span_getter = {"lookup_zones": True}
    doc = blank_nlp(text)
    ent = Span(doc, 10, 16, label="size")
    doc.spans["lookup_zones"] = [ent]
    doc = matcher(doc)

    assert len(doc.ents) == 2
    assert len(doc.spans["measurements"]) == 2
    assert [doc.ents[0].text, doc.ents[1].text] == ["2.0cm", "3cm"]
    assert [doc.ents[0]._.value.cm, doc.ents[1]._.value.cm] == [2.0, 3]
