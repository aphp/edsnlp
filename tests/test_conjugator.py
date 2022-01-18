import pytest
from mlconjug3 import Conjugator

from edsnlp.conjugator import conjugate, conjugate_verb, get_conjugated_verbs

pytestmark = pytest.mark.filterwarnings("ignore")


def test_conjugate_verb():

    conjugator = Conjugator("fr")

    tests = [
        (("aimer", "Indicatif", "Présent", "1s"), "aime"),
        (("aimer", "Indicatif", "Présent", "2s"), "aimes"),
        (("aimer", "Indicatif", "Présent", "1p"), "aimons"),
        (("aimer", "Indicatif", "Présent", "2p"), "aimez"),
        (("aimer", "Indicatif", "Présent", "3p"), "aiment"),
    ]

    verb = "aimer"

    df = conjugate_verb(verb, conjugator=conjugator)

    for (v, m, t, p), term in tests:
        row = df.query("verb == @v & mode == @m & tense == @t & person == @p").iloc[0]
        assert row.term == term


def test_conjugate():

    tests = [
        (("aimer", "Indicatif", "Présent", "1s"), "aime"),
        (("aimer", "Indicatif", "Présent", "2s"), "aimes"),
        (("aimer", "Indicatif", "Présent", "1p"), "aimons"),
        (("aimer", "Indicatif", "Présent", "2p"), "aimez"),
        (("aimer", "Indicatif", "Présent", "3p"), "aiment"),
        (("convaincre", "Indicatif", "Présent", "2s"), "convaincs"),
        (("convaincre", "Indicatif", "Présent", "1s"), "convaincs"),
        (("convaincre", "Indicatif", "Présent", "1p"), "convainquons"),
        (("convaincre", "Indicatif", "Présent", "2p"), "convainquez"),
        (("convaincre", "Indicatif", "Présent", "3p"), "convainquent"),
    ]

    conjugate("aimer")
    df = conjugate(["aimer", "convaincre"])

    for (v, m, t, p), term in tests:
        row = df.query("verb == @v & mode == @m & tense == @t & person == @p").iloc[0]
        assert row.term == term


def test_get_conjugated_verbs():

    terms = get_conjugated_verbs(
        ["aimer", "convaincre"],
        matches=[dict(mode="Indicatif", tense="Présent")],
    )

    get_conjugated_verbs(
        "aimer",
        matches=dict(mode="Indicatif", tense="Présent"),
    )

    assert set(terms) == {
        "aime",
        "aimes",
        "aimons",
        "aimez",
        "aiment",
        "convainc",
        "convaincs",
        "convainquons",
        "convainquez",
        "convainquent",
    }
