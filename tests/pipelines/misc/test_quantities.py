import pickle
from itertools import chain

import dill
import pytest
from pytest import fixture
from spacy.tokens.span import Span

import edsnlp
import edsnlp.pipes as eds
from edsnlp.core import PipelineProtocol

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
    model = edsnlp.blank("eds")
    model.add_pipe("eds.normalizer")
    model.add_pipe("eds.sentences")
    model.add_pipe("eds.tables")

    return model


def test_deprecated_pipe(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe("matcher", config=dict(terms={"patient": "patient"}))
    blank_nlp.add_pipe("eds.measurements")

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
        eds.quantities(quantities=["size", "weight", "bmi"], use_tables=True),
    )

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["quantities"]) == 15


def test_quantities_component(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    doc = blank_nlp(text)

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


def test_quantities_component_scaling(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    doc = blank_nlp(text)

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


def test_measure_label(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    doc = blank_nlp(text)

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


def test_quantities_all_input(blank_nlp: PipelineProtocol):
    all_text = "On mesure 13 mol/ml de ...On compte 16x10*9 ..."
    blank_nlp.add_pipe(eds.quantities(quantities="all", extract_ranges=True))

    doc = blank_nlp(all_text)

    m1, m2 = doc.spans["quantities"]

    assert str(m1._.value) == "13 mol_per_ml"
    assert str(m2._.value) == "16 x10*9"


def test_measure_str(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, res in [
        ("1m50", "1.5 m"),
        ("1,50cm", "1.5 cm"),
    ]:
        doc = blank_nlp(text)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_measure_repr(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
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

        assert repr(doc.spans["quantities"][0]._.value) == res


def test_compare(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    m1, m2 = "1m0", "120cm"
    m1 = blank_nlp(m1).spans["quantities"][0]
    m2 = blank_nlp(m2).spans["quantities"][0]
    assert m1._.value <= m2._.value
    assert m2._.value > m1._.value

    m3 = "Entre deux et trois metres"
    m4 = "De 2 à 3 metres"
    m3 = blank_nlp(m3).spans["quantities"][0]
    m4 = blank_nlp(m4).spans["quantities"][0]
    assert str(m3._.value) == "2-3 m"
    assert str(m4._.value) == "2-3 m"
    assert m4._.value.cm == (200.0, 300.0)

    assert m3._.value == m4._.value
    assert m3._.value <= m4._.value
    assert m3._.value >= m1._.value

    assert max(list(chain(m1._.value, m2._.value, m3._.value, m4._.value))).cm == 300


def test_unitless(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, res in [
        ("BMI: 24 .", "24 kg_per_m2"),
        ("Le patient mesure 1.5 ", "1.5 m"),
        ("Le patient mesure 152 ", "152 cm"),
        ("Le patient pèse 34 ", "34 kg"),
    ]:
        doc = blank_nlp(text)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_unitless_sequences(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, expected in [
        (
            "Poids (Kg) Taille (m) IMC\n57,0 1,70 22",
            [("weight", "57.0 kg"), ("size", "1.7 m"), ("bmi", "22 kg_per_m2")],
        ),
        (
            "poids / IMC : 57imc22 taille : 170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        (
            "poids / IMC : 57/22 taille : 170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        (
            "poids / IMC / taille : 57/22/150",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "150 cm")],
        ),
        (
            "poids / IMC / taille : 57 / 22 / 150",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "150 cm")],
        ),
        (
            "poids, taille, IMC : 57 et 170 et 22",
            [("weight", "57 kg"), ("size", "170 cm"), ("bmi", "22 kg_per_m2")],
        ),
        (
            "poids et IMC : 57 et 22 taille : 170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        (
            "poids - IMC : 57 - 22 taille : 170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        (
            "poids / IMC : 57 22 taille : 170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        (
            "poids / IMC :\t57\t22\n taille :\t170",
            [("weight", "57 kg"), ("bmi", "22 kg_per_m2"), ("size", "170 cm")],
        ),
        # ambiguous groups -> no match
        (
            "poids / truc / IMC : 57/3/22",
            [],
        ),
        (
            "poids / IMC / truc : 57/3/22",
            [],
        ),
        (
            "poids : 57/3/22",
            [],
        ),
    ]:
        doc = blank_nlp(text)

        assert [
            (span.label_, str(span._.value)) for span in doc.spans["quantities"]
        ] == expected


def test_unitless_sequences_are_dropped_when_ambiguous(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text in [
        "poids / truc / IMC : 57/3/22",
        "poids / IMC / truc : 57/3/22",
    ]:
        doc = blank_nlp(text)

        assert len(doc.spans["quantities"]) == 0


def test_non_matches(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text in [
        "On délivre à 10 g / h.",
        "Le patient grandit de 10 cm par jour ",
        "Truc 10cma truc",
        "01.42.43.56.78 m",
    ]:
        doc = blank_nlp(text)

        assert len(doc.spans["quantities"]) == 0


def test_numbers(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, res in [
        ("deux m", "2 m"),
        ("2 m", "2 m"),
        ("⅛ m", "0.125 m"),
        ("0 m", "0 m"),
        ("55 @ 77777 cm", "77777 cm"),
    ]:
        doc = blank_nlp(text)

        assert str(doc.spans["quantities"][0]._.value) == res


def test_ranges(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, res, snippet in [
        ("Le patient fait entre 1 et 2m", "1-2 m", "entre 1 et 2m"),
        ("On mesure de 2 à 2.5 dl d'eau", "2-2.5 dl", "de 2 à 2.5 dl"),
    ]:
        doc = blank_nlp(text)

        quantity = doc.spans["quantities"][0]
        assert str(quantity._.value) == res
        assert quantity.text == snippet


def test_operator(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(eds.quantities(quantities="all"))

    doc = blank_nlp("< 5 µl et supérieur à 8 ui")
    quantities = doc.spans["quantities"]

    assert [
        (span.label_, str(span._.value), span._.value.operator) for span in quantities
    ] == [
        ("µl", "<5 µl", "<"),
        ("ui", ">8 ui", ">"),
    ]


def test_quantities_string_config_without_tables_pipe():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.quantities(quantities="weight", use_tables=True))

    doc = nlp("Poids : 65")

    assert [str(span._.value) for span in doc.spans["quantities"]] == ["65 kg"]


def test_valueless_patterns(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities={
                "status": {
                    "unit": "bool",
                    "valueless_patterns": [
                        {
                            "terms": ["positif", "positive"],
                            "quantity": {"value": 1, "unit": "bool"},
                        },
                        {
                            "regex": [r"n[eé]gati(?:f|ve)s?"],
                            "quantity": {"value": 0, "unit": "bool"},
                        },
                    ],
                }
            }
        ),
    )

    doc = blank_nlp("Résultat positif puis négatif")

    assert [(span.text, str(span._.value)) for span in doc.spans["quantities"]] == [
        ("positif", "1 bool"),
        ("négatif", "0 bool"),
    ]


def test_table_unit_linking(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities={"mass_col": {"unit": "mg"}, "vol_col": {"unit": "ml"}},
            use_tables=True,
        )
    )

    doc = blank_nlp(
        "mg | 5 | mL | 0.3\n"
        "mg | 7 | mL | 0.4\n"
    )  # fmt: skip

    assert [str(span._.value) for span in doc.spans["quantities"]] == [
        "5 mg",
        "0.3 ml",
        "7 mg",
        "0.4 ml",
    ]


def test_table_unit_and_power_linking(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities="all",
            use_tables=True,
        )
    )

    doc = blank_nlp(
        "Value | Power | Unit\n"
        "4.2 | x10*3 | g/L\n"
    )  # fmt: skip

    assert [
        (span.text, span.label_, str(span._.value), span._.value.g_per_l)
        for span in doc.spans["quantities"]
    ] == [
        ("4.2", "x10*3_g_per_l", "4.2 x10*3_g_per_l", 4200.0),
    ]


def test_table_unit_linking_tie_breaker(blank_nlp: PipelineProtocol):
    text = "mg | 5 | mL\nmg | 7 | mL\n"

    blank_nlp.add_pipe(
        eds.quantities(
            quantities={"mass_col": {"unit": "mg"}, "vol_col": {"unit": "ml"}},
            use_tables=True,
            prefer_measure_before_unit=True,
        )
    )
    doc = blank_nlp(text)
    assert [str(span._.value) for span in doc.spans["quantities"]] == ["5 mg", "7 mg"]

    blank_nlp_2 = edsnlp.blank("eds")
    blank_nlp_2.add_pipe(eds.normalizer())
    blank_nlp_2.add_pipe(eds.sentences())
    blank_nlp_2.add_pipe(eds.tables())
    blank_nlp_2.add_pipe(
        eds.quantities(
            quantities={"mass_col": {"unit": "mg"}, "vol_col": {"unit": "ml"}},
            use_tables=True,
            prefer_measure_before_unit=False,
        ),
    )
    doc = blank_nlp_2(text)
    assert [str(span._.value) for span in doc.spans["quantities"]] == ["5 ml", "7 ml"]


def test_merge_align(blank_nlp):
    blank_nlp.add_pipe(
        eds.quantities(
            extract_ranges=True,
            use_tables=True,
            merge_mode="align",
            span_getter={"candidates": True},
            span_setter={"ents": True},
        )
    )
    doc = blank_nlp.make_doc(text)
    ent = Span(doc, 10, 15, label="size")
    doc.spans["candidates"] = [ent]
    doc = blank_nlp(doc)

    assert len(doc.ents) == 1
    assert str(ent._.value) == "2.0 cm"


def test_merge_intersect(blank_nlp):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    pipe = blank_nlp.pipes.quantities
    pipe.merge_mode = "intersect"
    pipe.span_setter = {**pipe.span_setter, "ents": True}
    pipe.span_getter = {"lookup_zones": True}
    doc = blank_nlp.make_doc(text)
    ent = Span(doc, 10, 16, label="size")
    doc.spans["lookup_zones"] = [ent]
    doc = blank_nlp(doc)

    assert len(doc.ents) == 2
    assert len(doc.spans["quantities"]) == 2
    assert [doc.ents[0].text, doc.ents[1].text] == ["2.0cm", "3cm"]
    assert [doc.ents[0]._.value.cm, doc.ents[1]._.value.cm] == [2.0, 3]


def test_quantity_snippets(blank_nlp):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    for text, result in [
        ("0.50g", ["0.5 g"]),
        ("0.050g", ["0.05 g"]),
        ("1 m 50", ["1.5 m"]),
        ("1.50 m", ["1.5 m"]),
        ("1,50m", ["1.5 m"]),
        ("57/22 kg", ["2.590909090909091 kg"]),
        ("2.0cm x 3cm", ["2.0 cm", "3 cm"]),
        ("2 par 1mm", ["2 mm", "1 mm"]),
        ("8, 13 et 15dm", ["8 dm", "13 dm", "15 dm"]),
        ("1 / 50  kg", ["0.02 kg"]),
    ]:
        doc = blank_nlp(text)

        assert [str(span._.value) for span in doc.spans["quantities"]] == result


def test_error_management(blank_nlp):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
    text = (
        "Leucocytes ¦ ¦ ¦4.2 ¦ ¦4.0-10.0\n"
        "Hémoglobine ¦ ¦9.0 - ¦ ¦13-14\n"
    )  # fmt: skip
    doc = blank_nlp(text)

    assert len(doc.spans["quantities"]) == 0


def test_conversions(blank_nlp):
    blank_nlp.add_pipe(eds.quantities(extract_ranges=True, use_tables=True))
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
        result = getattr(doc.spans["quantities"][0]._.value, unit)
        assert result == pytest.approx(expected, 1e-6), (
            f"{result} != {expected} for {text} in {unit}"
        )


def test_time_quantities(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(quantities={"duration": {"unit": "second"}}),
    )
    tests = [
        ("Le test a duré entre 5'14'' et 6'05.", (5 * 60 + 14, 6 * 60 + 5)),  # noqa: E501  # fmt: skip
        ("La perfusion a duré 2 heures.", (2 * 3600,)),
        ("L'examen a pris 45 min.", (45 * 60,)),
        ("La procédure a duré 1h30.", (1 * 3600 + 30 * 60,)),
        ("Le patient a attendu 90 secondes.", (90,)),
    ]

    for text, expected_seconds in tests:
        doc = blank_nlp(text)
        quantities = sorted(doc.spans["quantities"])
        for i, expected in enumerate(expected_seconds):
            value = quantities[i]._.value
            seconds = value.second
            assert seconds == pytest.approx(expected, 1e-6)


@pytest.mark.parametrize("pickler_module", ["pickle", "dill"])
def test_quantities_pickle_dump_and_load(
    blank_nlp: PipelineProtocol,
    pickler_module: str,
):
    pickler = pickle if pickler_module == "pickle" else dill

    blank_nlp.add_pipe(
        eds.quantities(quantities={"duration": {"unit": "second"}}, use_tables=False),
    )

    doc = blank_nlp("La procédure a duré 1h30.")
    value = doc.spans["quantities"][0]._.value

    reloaded_value = pickler.loads(pickler.dumps(value))
    assert reloaded_value.minute == pytest.approx(90.0, 1e-6)

    reloaded_doc = pickler.loads(pickler.dumps(doc))
    assert reloaded_doc.spans["quantities"][0]._.value.minute == pytest.approx(
        90.0, 1e-6
    )


def test_complex_table_quantities_parsing(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities={
                "mass_concentration": {"unit": "mg_per_l"},
                "urine_volume": {"unit": "ml"},
                "weight": {"unit": "kg"},
                "size": {"unit": "m"},
                "status": {
                    "unit": "bool",
                    "valueless_patterns": [
                        {
                            "terms": ["positif"],
                            "quantity": {"value": 1, "unit": "bool"},
                        },
                        {
                            "terms": ["negatif"],
                            "quantity": {"value": 0, "unit": "bool"},
                        },
                    ],
                },
            },
            use_tables=True,
        ),
    )

    text = (
        "Analyse | Statut | Valeur | Unite | Commentaire\n"
        "CRP | positif | > 5 | mg/L | controle demain\n"
        "Volume urine | negatif | 0.3 | mL | a surveiller\n"
        "Poids | stable | 67 | kg | ok\n"
        "Taille | notee | 1.68 | m | mesure manuelle\n"
        "Commentaire | en hausse | controle | - | non quantitatif\n"
    )  # fmt: skip

    doc = blank_nlp(text)

    assert [
        (span.text, span.label_, str(span._.value), span._.value.operator)
        for span in doc.spans["quantities"]
    ] == [
        ("positif", "status", "1 bool", "="),
        ("negatif", "status", "0 bool", "="),
        ("> 5", "mass_concentration", ">5 mg_per_l", ">"),
        ("0.3", "urine_volume", "0.3 ml", "="),
        ("67", "weight", "67 kg", "="),
        ("1.68", "size", "1.68 m", "="),
    ]


def test_multiple_tables_and_multi_quantities_per_row(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities={
                "mass_concentration": {"unit": "mg_per_l"},
                "urine_volume": {"unit": "ml"},
                "weight": {"unit": "kg"},
                "size": {"unit": "m"},
            },
            use_tables=True,
        ),
    )

    text = (
        "Analyse | Valeur | Unite | Valeur2 | Unite2\n"
        "CRP | 5 | mg/L | 0.3 | mL\n"
        "\n"
        "Analyse | Resultats\n"
        "Bilan | 7 mg/L ; 0.4 mL\n"
        "\n"
        "Mesure | Valeur\n"
        "Poids | 67 kg\n"
        "Taille | 1.68 m\n"
    )  # fmt: skip

    doc = blank_nlp(text)

    assert len(doc.spans["tables"]) == 3
    assert [
        (span.text, span.label_, str(span._.value)) for span in doc.spans["quantities"]
    ] == [
        ("5", "mass_concentration", "5 mg_per_l"),
        ("0.3", "urine_volume", "0.3 ml"),
        ("7 mg/L", "mass_concentration", "7 mg_per_l"),
        ("0.4 mL", "urine_volume", "0.4 ml"),
        ("67 kg", "weight", "67 kg"),
        ("1.68 m", "size", "1.68 m"),
    ]


def test_table_header_units(blank_nlp: PipelineProtocol):
    blank_nlp.add_pipe(
        eds.quantities(
            quantities={
                "weight": {"unit": "kg"},
                "size": {"unit": "m"},
                "bmi": {"unit": "kg_per_m2"},
                "duration": {"unit": "second"},
            },
            use_tables=True,
        ),
    )

    text = (
        "Patient | Poids (kg) | Taille (m) | IMC (kg/m2)\n"
        "A | 67 | 1.68 | 23.7\n"
        "\n"
        "Quantity | Unit | Measurement duration (s)\n"
        "150 | cm | 5\n"
        "55 | kg | 10\n"
    )  # fmt: skip

    doc = blank_nlp(text)

    assert len(doc.spans["tables"]) == 2
    assert [
        (span.text, span.label_, str(span._.value)) for span in doc.spans["quantities"]
    ] == [
        ("67", "weight", "67 kg"),
        ("1.68", "size", "1.68 m"),
        ("23.7", "bmi", "23.7 kg_per_m2"),
        ("150", "size", "150 cm"),
        ("5", "duration", "5 second"),
        ("55", "weight", "55 kg"),
        ("10", "duration", "10 second"),
    ]
