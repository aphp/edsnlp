import pytest
from pytest import mark

import edsnlp
import edsnlp.pipes as eds
from edsnlp.pipes.core.sentences.sentences import (
    DEFAULT_CAPITALIZED_SHAPES,
    LEGACY_CAPITALIZED_SHAPES,
    generate_capitalized_shapes,
)

text = (
    "Le patient est admis pour des douleurs dans le bras droit. "
    "mais n'a pas de problème de locomotion. \n"
    "Historique d'AVC dans la famille\n"
    "Mais ne semble pas en être un\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit !"
    "Il est contaminé à E.Coli? c'est un problème, il faut s'en occuper."
)


@mark.parametrize("endlines", [True, False])
def test_sentences(endlines):
    nlp = edsnlp.blank("fr")
    nlp.add_pipe("eds.sentences", config={"use_endlines": endlines})
    doc = nlp.make_doc(text)

    if endlines:
        doc[28].tag_ = "EXCLUDED"

    doc = nlp(doc)

    sents_text = [sent.text for sent in doc.sents]

    if endlines:
        assert sents_text == [
            "Le patient est admis pour des douleurs dans le bras droit.",
            "mais n'a pas de problème de locomotion. \n",
            "Historique d'AVC dans la famille\nMais ne semble pas en être un\n",
            "Pourrait être un cas de rhume.\n",
            "Motif :\n",
            "Douleurs dans le bras droit !",
            "Il est contaminé à E.Coli?",
            "c'est un problème, il faut s'en occuper.",
        ]
    else:
        assert sents_text == [
            "Le patient est admis pour des douleurs dans le bras droit.",
            "mais n'a pas de problème de locomotion. \n",
            "Historique d'AVC dans la famille\n",
            "Mais ne semble pas en être un\n",
            "Pourrait être un cas de rhume.\n",
            "Motif :\n",
            "Douleurs dans le bras droit !",
            "Il est contaminé à E.Coli?",
            "c'est un problème, il faut s'en occuper.",
        ]

    nlp("")


def test_false_positives(blank_nlp):
    false_positives = [
        "02.04.2018",
        "E.Coli",
    ]

    for fp in false_positives:
        doc = blank_nlp(fp)
        assert len(list(doc.sents)) == 1


@pytest.mark.parametrize(
    "text",
    [
        "10.10.2010:RCP",
        "10.10.2010 : RCP",
        "02.04.2018 : RCP",
        "10/10/2010 : RCP",
    ],
)
def test_false_positives_dotted_dates_with_labels(blank_nlp, text):
    doc = blank_nlp(text)
    assert [sent.text for sent in doc.sents] == [text]


def test_newlines_double():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.sentences",
        config={
            "punct_chars": [],
            "ignore_excluded": False,
            "check_capitalized": False,
            "min_newline_count": 2,
            "hard_newline_count": None,
        },
    )

    doc = nlp(
        """\
Lundi
Mardi
Mercredi
Le patient est admis. Des douleurs dans le bras droit
\n\n
jeudi."""
    )
    assert len(list(doc.sents)) == 2

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.sentences",
        config={
            "punct_chars": [],
            "ignore_excluded": False,
            "check_capitalized": True,
            "min_newline_count": 2,
            "hard_newline_count": None,
        },
    )

    doc = nlp(
        """\
Lundi
Mardi
Mercredi
Le patient est admis. Des douleurs dans le bras droit
\n
jeudi."""
    )
    assert len(list(doc.sents)) == 1


def test_hard_newlines_force_split_before_date():
    # https://github.com/aphp/edsnlp/issues/277
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences(hard_newline_count=2))

    doc = nlp(
        """\
ANTECEDANT

15/03/2020 Antécédant 1
v antecedant numero 2
"""
    )

    assert [sent.text for sent in doc.sents] == [
        "ANTECEDANT\n\n",
        "15/03/2020 Antécédant 1\nv antecedant numero 2\n",
    ]


def test_sentences_bullet_starters():
    """
    FR language doesn't split punctuations so '--' is a full token
    and not treated as a bullet starter
    """
    nlp = edsnlp.blank("fr")
    nlp.add_pipe(
        edsnlp.pipes.sentences(
            use_bullet_start=True,
            bullet_starters=["-"],
        )
    )

    text = (
        "Symptômes observés:\n"
        "- Douleur thoracique\n"
        "-- forte toux\n"
        "- Fièvre élevée\n"
        "- Toux sèche\n"
        "Le patient semble stable - pas d'évolution\n"
    )

    doc = nlp(text)
    sents_text = [sent.text for sent in doc.sents]

    assert sents_text == [
        "Symptômes observés:\n",
        "- Douleur thoracique\n-- forte toux\n",
        "- Fièvre élevée\n",
        "- Toux sèche\n",
        "Le patient semble stable - pas d'évolution\n",
    ]


def test_sentences_bullet_edge_cases():
    """Test edge cases for bullet starters"""
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        edsnlp.pipes.sentences(
            use_bullet_start=True,
            bullet_starters=["-"],
        )
    )

    text1 = "Le patient - âgé de 45 ans - présente des symptômes."
    doc1 = nlp(text1)
    assert len(list(doc1.sents)) == 1

    text2 = "Symptômes:   \n- Fièvre\t\n- Toux"
    doc2 = nlp(text2)
    sents2 = [sent.text for sent in doc2.sents]
    assert sents2 == ["Symptômes:   \n", "- Fièvre\t\n", "- Toux"]

    text3 = "Item:\n_ Premier point\n_ Deuxième point"
    doc3 = nlp(text3)
    sents3 = [sent.text for sent in doc3.sents]
    assert len(sents3) == 1


def test_sentences_multiple_bullet_types():
    """Test multiple bullet starter types"""
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(edsnlp.pipes.sentences(use_bullet_start=True))

    text = "Liste mixte:\n- Point A\n* Point B\n• Point C\n· Point D"
    doc = nlp(text)
    assert len(list(doc.sents)) == 5  # header + 4 bullets


def make_nlp(
    cap_shapes=None,
    mode: str | None = None,
    check_capitalized: bool = True,
    use_bullet_start: bool = True,
    hard_newline_count: int | None = 2,
):
    nlp = edsnlp.blank("eds")
    config = {
        "use_bullet_start": use_bullet_start,
        "bullet_starters": ["-"],
        "check_capitalized": check_capitalized,
        "capitalized_shapes": cap_shapes,
        "hard_newline_count": hard_newline_count,
    }
    if mode is not None:
        config["capitalized_mode"] = mode
    nlp.add_pipe("eds.sentences", config=config)
    return nlp


def test_all_caps_sections_expanded_mode():
    nlp = make_nlp(mode="expanded")
    doc = nlp("CONCLUSION\nSuite\n")
    assert [s.text for s in doc.sents] == ["CONCLUSION\n", "Suite\n"]


def test_all_caps_with_bullets_expanded_mode():
    nlp = make_nlp(mode="expanded")
    text = "EVOLUTION\n- Fièvre\n- Toux\n"
    assert [s.text for s in nlp(text).sents] == [
        "EVOLUTION\n",
        "- Fièvre\n",
        "- Toux\n",
    ]


def test_custom_shapes_override_titlecase_only():
    nlp = make_nlp(cap_shapes=["Xxxxx"])
    doc = nlp("Titre\nSuite\n")
    assert [s.text for s in doc.sents] == ["Titre\n", "Suite\n"]


def test_disable_capitalized_rule_keeps_bullets_only():
    nlp = make_nlp(check_capitalized=False)
    text = "CONCLUSION\n- Fièvre\n- Toux\n"
    sents = [s.text for s in nlp(text).sents]
    assert "- Fièvre\n" in sents and "- Toux\n" in sents


@pytest.mark.parametrize(
    "mode, expected",
    [
        (
            "legacy",
            [
                "Une première phrase.",
                "Une deuxième\n",
                "Peut-être un autre\nET encore une.",
            ],
        ),
        (
            "expanded",
            [
                "Une première phrase.",
                "Une deuxième\n",
                "Peut-être un autre\n",
                "ET encore une.",
            ],
        ),
    ],
)
def test_old_newline_split_behavior_mapped_to_current_modes(mode, expected):
    # Adapted from the old split_on_newlines test:
    # - with_capitalized -> capitalized_mode="legacy"
    # - with_uppercase -> capitalized_mode="expanded"
    nlp = make_nlp(mode=mode, use_bullet_start=False)
    doc = nlp("Une première phrase. Une deuxième\nPeut-être un autre\nET encore une.")
    assert [s.text for s in doc.sents] == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ÉTAT CIVIL  \nSuite\n", ["ÉTAT CIVIL  \n", "Suite\n"]),
        ("CONCLUSION\r\n- Fièvre\r\n", ["CONCLUSION\r\n", "- Fièvre\r\n"]),
    ],
)
def test_newline_robustness_with_expanded_mode(text, expected):
    nlp = make_nlp(mode="expanded")
    doc = nlp(text)
    assert [s.text for s in doc.sents] == expected


def test_legacy_mode_behavior_non_regression():
    nlp = make_nlp(mode="legacy", hard_newline_count=None)
    doc = nlp("hémoculture\n\nCONCLUSION\nSuite\n")
    assert [s.text for s in doc.sents] == ["hémoculture\n\nCONCLUSION\n", "Suite\n"]


def test_generate_returns_tuple_of_strings_and_no_duplicates():
    shapes = generate_capitalized_shapes()
    assert isinstance(shapes, tuple)
    assert all(isinstance(s, str) for s in shapes)
    assert len(shapes) == len(set(shapes))


def test_toggles_all_caps_titlecase_apostrophe():
    s_all = generate_capitalized_shapes(
        include_all_caps=True,
        include_titlecase=True,
        include_apostrophe=True,
    )
    assert "XX" in s_all and "Xx" in s_all and "X'" in s_all

    s_no_caps = generate_capitalized_shapes(
        include_all_caps=False,
        include_titlecase=True,
        include_apostrophe=True,
    )
    assert "XX" not in s_no_caps and "Xx" in s_no_caps and "X'" in s_no_caps

    s_no_title = generate_capitalized_shapes(
        include_all_caps=True,
        include_titlecase=False,
        include_apostrophe=True,
    )
    assert "XX" in s_no_title and "Xx" not in s_no_title and "X'" in s_no_title

    s_no_apo = generate_capitalized_shapes(
        include_all_caps=True,
        include_titlecase=True,
        include_apostrophe=False,
    )
    assert "XX" in s_no_apo and "Xx" in s_no_apo and "X'" not in s_no_apo

    s_none = generate_capitalized_shapes(
        include_all_caps=False,
        include_titlecase=False,
        include_apostrophe=False,
    )
    assert s_none == tuple()


def test_defaults_match_default_constant():
    expected = generate_capitalized_shapes(
        upper_min=2,
        upper_max=13,
        x_min=2,
        x_max=12,
        include_apostrophe=True,
    )
    assert DEFAULT_CAPITALIZED_SHAPES == expected


def test_legacy_exact_values():
    assert LEGACY_CAPITALIZED_SHAPES == ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx")


def test_bounds_min_max_presence():
    shapes = generate_capitalized_shapes(
        upper_min=2,
        upper_max=4,
        x_min=2,
        x_max=3,
        include_apostrophe=False,
    )
    assert "XX" in shapes and "XXXX" in shapes
    assert "Xx" in shapes and "Xxx" in shapes
