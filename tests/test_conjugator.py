from edsnlp.conjugator import normalize, conjugate_verb, conjugate, get_conjugated_verbs


def test_normalize():

    t1 = ("Mode", "Tense", "Term")
    t2 = ("Mode", "Tense", "Person", "Term")

    c1 = normalize(t1)
    c2 = normalize(t2)

    assert c1.mode == c2.mode == "Mode"
    assert c1.tense == c2.tense == "Tense"
    assert c1.person is None
    assert c2.person == "Person"
    assert c1.term == c2.term == "Term"


def test_conjugate_verb():

    tests = [
        (("aimer", "Indicatif", "Présent", "1s"), "aime"),
        (("aimer", "Indicatif", "Présent", "2s"), "aimes"),
        (("aimer", "Indicatif", "Présent", "1p"), "aimons"),
        (("aimer", "Indicatif", "Présent", "2p"), "aimez"),
        (("aimer", "Indicatif", "Présent", "3p"), "aiment"),
    ]

    verb = "aimer"

    df = conjugate_verb(verb)

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

    df = conjugate(["aimer", "convaincre"])

    for (v, m, t, p), term in tests:
        row = df.query("verb == @v & mode == @m & tense == @t & person == @p").iloc[0]
        assert row.term == term


def test_get_conjugated_verbs():

    terms = get_conjugated_verbs(
        ["aimer", "convaincre"],
        matches=[dict(mode="Indicatif", tense="Présent")],
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
