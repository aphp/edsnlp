from itertools import chain

import pytest
import spacy
from pytest import fixture, raises
from spacy.tokens.span import Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.misc.quantities import QuantitiesMatcher

text = (
    "Le patient fait 1 m 50 kg. La tumeur fait 2.0cm x 3cm. \n"
    "Une autre tumeur plus petite fait 2 par 1mm.\n"
    "Les trois éléments font 8, 13 et 15dm.\n"
    """
    Leucocytes ¦mm ¦ ¦4.2 ¦ ¦4.0-10.0
    Hémoglobine ¦ ¦9.0 - ¦ g ¦13-14
    Hémoglobine ¦ ¦9.0 - ¦ ¦ xxx
    """
)


@fixture
def blank_nlp():
    model = spacy.blank("eds")
    model.add_pipe("eds.normalizer")
    model.add_pipe("eds.sentences")
    model.add_pipe("eds.tables")

    return model


@fixture
def matcher(blank_nlp):
    matcher = QuantitiesMatcher(blank_nlp, extract_ranges=True, use_tables=True)
    return matcher


def test_deprecated_pipe(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.measurements",
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["quantities"]) == 15
    assert len(doc.spans["measurements"]) == 15


def test_deprecated_arg(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.measurements", config=dict(measurements=["size", "weight", "bmi"])
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["quantities"]) == 15
    assert len(doc.spans["measurements"]) == 15


def test_default_factory(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe(
        "eds.quantities",
        config={"quantities": ["size", "weight", "bmi"], "use_tables": True},
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["quantities"]) == 15


def test_quantities_component(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["quantities"]

    doc = matcher(doc)

    for span_key in ["quantities", "measurements"]:
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = doc.spans[span_key]

        assert str(m1._.value) == "1 m"
        assert str(m2._.value) == "50 kg"
        assert str(m3._.value) == "2.0 cm"
        assert str(m4._.value) == "3 cm"
        assert str(m5._.value) == "2 mm"
        assert str(m6._.value) == "1 mm"
        assert str(m7._.value) == "8 dm"
        assert str(m8._.value) == "13 dm"
        assert str(m9._.value) == "15 dm"
        assert str(m10._.value) == "4.2 mm"
        assert str(m11._.value) == "4.0-10.0 mm"
        assert str(m12._.value) == "9.0 g"
        assert str(m13._.value) == "13-14 g"


def test_quantities_component_scaling(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["quantities"]

    doc = matcher(doc)

    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = doc.spans["quantities"]

    assert abs(m1._.value.cm - 100) < 1e-6
    assert abs(m2._.value.mg - 50000000.0) < 1e-6
    assert abs(m3._.value.mm - 20) < 1e-6
    assert abs(m4._.value.mm - 30) < 1e-6
    assert abs(m5._.value.cm - 0.2) < 1e-6
    assert abs(m6._.value.cm - 0.1) < 1e-6
    assert abs(m7._.value.dm - 8.0) < 1e-6
    assert abs(m8._.value.m - 1.3) < 1e-6
    assert abs(m9._.value.m - 1.5) < 1e-6
    assert abs(m10._.value.mm - 4.2) < 1e-6
    assert abs(m11._.value.mm[0] - 4.0) < 1e-6
    assert abs(m11._.value.mm[1] - 10.0) < 1e-6
    assert abs(m12._.value.g - 9) < 1e-6
    assert abs(m13._.value.g[0] - 13.0) < 1e-6
    assert abs(m13._.value.g[1] - 14.0) < 1e-6


def test_measure_label(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    doc = blank_nlp(text)
    doc = matcher(doc)

    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = doc.spans["quantities"]

    assert m1.label_ == "size"
    assert m2.label_ == "weight"
    assert m3.label_ == "size"
    assert m4.label_ == "size"
    assert m5.label_ == "size"
    assert m6.label_ == "size"
    assert m7.label_ == "size"
    assert m8.label_ == "size"
    assert m9.label_ == "size"
    assert m10.label_ == "size"
    assert m11.label_ == "size"
    assert m12.label_ == "weight"
    assert m13.label_ == "weight"


def test_quantities_all_input(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    all_text = "On mesure 13 mol/ml de ..." "On compte 16x10*9 ..."
    blank_nlp.add_pipe(
        "eds.quantities",
        config={"quantities": "all", "extract_ranges": True},
    )

    doc = blank_nlp(all_text)

    m1, m2 = doc.spans["quantities"]

    assert str(m1._.value) == "13 mol_per_ml"
    assert str(m2._.value) == "16 x10*9"


def test_measure_str(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text, res in [
        ("1m50", "1.5 m"),
        ("1,50cm", "1.5 cm"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_measure_repr(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text, res in [
        (
            "1m50",
            "Quantity(1.5, 'm')",
        ),
        (
            "1,50cm",
            "Quantity(1.5, 'cm')",
        ),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert repr(doc.spans["quantities"][0]._.value) == res


def test_compare(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    m1, m2 = "1m0", "120cm"
    m1 = matcher(blank_nlp(m1)).spans["quantities"][0]
    m2 = matcher(blank_nlp(m2)).spans["quantities"][0]
    assert m1._.value <= m2._.value
    assert m2._.value > m1._.value

    m3 = "Entre deux et trois metres"
    m4 = "De 2 à 3 metres"
    m3 = matcher(blank_nlp(m3)).spans["quantities"][0]
    m4 = matcher(blank_nlp(m4)).spans["quantities"][0]
    assert str(m3._.value) == "2-3 m"
    assert str(m4._.value) == "2-3 m"
    assert m4._.value.cm == (200.0, 300.0)

    assert m3._.value == m4._.value
    assert m3._.value <= m4._.value
    assert m3._.value >= m1._.value

    assert max(list(chain(m1._.value, m2._.value, m3._.value, m4._.value))).cm == 300


def test_unitless(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text, res in [
        ("BMI: 24 .", "24 kg_per_m2"),
        ("Le patient mesure 1.5 ", "1.5 m"),
        ("Le patient mesure 152 ", "152 cm"),
        ("Le patient pèse 34 ", "34 kg"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_non_matches(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text in [
        "On délivre à 10 g / h.",
        "Le patient grandit de 10 cm par jour ",
        "Truc 10cma truc",
        "01.42.43.56.78 m",
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert len(doc.spans["quantities"]) == 0


def test_numbers(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text, res in [
        ("deux m", "2 m"),
        ("2 m", "2 m"),
        ("⅛ m", "0.125 m"),
        ("0 m", "0 m"),
        ("55 @ 77777 cm", "77777 cm"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_ranges(
    blank_nlp: PipelineProtocol,
    matcher: QuantitiesMatcher,
):
    for text, res, snippet in [
        ("Le patient fait entre 1 et 2m", "1-2 m", "entre 1 et 2m"),
        ("On mesure de 2 à 2.5 dl d'eau", "2-2.5 dl", "de 2 à 2.5 dl"),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        quantity = doc.spans["quantities"][0]
        assert str(quantity._.value) == res
        assert quantity.text == snippet


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


def test_merge_intersect(blank_nlp, matcher: QuantitiesMatcher):
    matcher.merge_mode = "intersect"
    matcher.span_setter = {**matcher.span_setter, "ents": True}
    matcher.span_getter = {"lookup_zones": True}
    doc = blank_nlp(text)
    ent = Span(doc, 10, 16, label="size")
    doc.spans["lookup_zones"] = [ent]
    doc = matcher(doc)

    assert len(doc.ents) == 2
    assert len(doc.spans["quantities"]) == 2
    assert [doc.ents[0].text, doc.ents[1].text] == ["2.0cm", "3cm"]
    assert [doc.ents[0]._.value.cm, doc.ents[1]._.value.cm] == [2.0, 3]


def test_quantity_snippets(blank_nlp, matcher: QuantitiesMatcher):
    for text, result in [
        ("0.50g", ["0.5 g"]),
        ("0.050g", ["0.05 g"]),
        ("1 m 50", ["1.5 m"]),
        ("1.50 m", ["1.5 m"]),
        ("1,50m", ["1.5 m"]),
        ("2.0cm x 3cm", ["2.0 cm", "3 cm"]),
        ("2 par 1mm", ["2 mm", "1 mm"]),
        ("8, 13 et 15dm", ["8 dm", "13 dm", "15 dm"]),
        ("1 / 50  kg", ["0.02 kg"]),
    ]:
        doc = blank_nlp(text)
        doc = matcher(doc)

        assert [str(span._.value) for span in doc.spans["quantities"]] == result


def test_error_management(blank_nlp, matcher: QuantitiesMatcher):
    text = """
        Leucocytes ¦ ¦ ¦4.2 ¦ ¦4.0-10.0
        Hémoglobine ¦ ¦9.0 - ¦ ¦13-14
        """
    doc = blank_nlp(text)
    doc = matcher(doc)

    assert len(doc.spans["quantities"]) == 0


def test_conversions(blank_nlp, matcher: QuantitiesMatcher):
    tests = [
        ("20 dm3", "l", 20),
        ("20 dm3", "m3", 0.02),
        ("20 dm3", "cm3", 20000),
        ("10 l", "cm3", 10000),
        ("10 l", "cl", 1000),
        ("25 kg/m2", "kg_per_cm2", 0.0025),
        ("2.4 x10*9µl", "l", 2400),
    ]

    for text, unit, expected in tests:
        doc = blank_nlp(text)
        doc = matcher(doc)
        result = getattr(doc.spans["quantities"][0]._.value, unit)
        assert result == pytest.approx(
            expected, 1e-6
        ), f"{result} != {expected} for {text} in {unit}"
