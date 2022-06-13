import spacy
from pytest import fixture, raises
from spacy.language import Language

from edsnlp.pipelines.misc.measurements import MeasurementsMatcher
from edsnlp.pipelines.misc.measurements.factory import DEFAULT_CONFIG

text = (
    "Le patient fait 1 m 50 kg. La tumeur fait 2cm x 3cm. \n"
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
        **DEFAULT_CONFIG,
    )


def test_default_factory(blank_nlp: Language):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.measurements",
        config=dict(measurements=["eds.size", "eds.weight", "eds.bmi"]),
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

    assert str(m1._.value) == "1.0 m"
    assert str(m2._.value) == "50.0 kg"
    assert str(m3._.value) == "2.0 cm"
    assert str(m4._.value) == "3.0 cm"
    assert str(m5._.value) == "2.0 mm"
    assert str(m6._.value) == "1.0 mm"
    assert str(m7._.value) == "8.0 dm"
    assert str(m8._.value) == "13.0 dm"
    assert str(m9._.value) == "15.0 dm"


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

    assert m1.label_ == "eds.size"
    assert m2.label_ == "eds.weight"
    assert m3.label_ == "eds.size"
    assert m4.label_ == "eds.size"
    assert m5.label_ == "eds.size"
    assert m6.label_ == "eds.size"
    assert m7.label_ == "eds.size"
    assert m8.label_ == "eds.size"
    assert m9.label_ == "eds.size"


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


def test_unitless(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text, res in [
        ("BMI: 24 .", "24.0 kg_per_m2"),
        ("Le patient mesure 1.5 ", "1.5 m"),
        ("Le patient mesure 152 ", "152.0 cm"),
        ("Le patient pèse 34 ", "34.0 kg"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["measurements"][0]._.value) == res


def test_non_matches(blank_nlp: Language, matcher: MeasurementsMatcher):
    for text in [
        "On délivre à 10 g / h.",
        "Le patient grandit de 10 cm par jour ",
        "Truc 10cma truc",
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert len(doc.spans["measurements"]) == 0
