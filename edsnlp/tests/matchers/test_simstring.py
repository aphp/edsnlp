from pytest import mark

from edsnlp.matchers.simstring import SimstringMatcher


def test_simstring_matcher(doc, nlp):
    matcher = SimstringMatcher(nlp.vocab, attr="TEXT")

    matcher.build_patterns(
        nlp,
        {
            "patient": ["patient"],
            "locomotion": ["locomotions"],
        },
    )

    matches = [m.text for m in matcher(doc, as_spans=True)]

    assert matches == ["patient", "locomotion", "patient"]


def test_with_normalizer(blank_nlp):
    blank_nlp.add_pipe("eds.normalizer")
    pattern = "matching"
    matcher = SimstringMatcher(
        blank_nlp.vocab,
        attr="NORM",
        threshold=0.75,
        measure="dice",
        ignore_space_tokens=True,
        ignore_excluded=True,
    )

    matcher.build_patterns(
        blank_nlp,
        {
            "test": [pattern],
            "C220": ["carcinome hépatocellulaire", "carc. hépatocellulaire"],
            "N02BE01": ["paracetamol"],
        },
    )

    texts = (
        ("Ceci est un test de matching", ["matching"]),
        ("Ceci est un test de matchings", ["matchings"]),
        ("On prescrit du paracétomol      , un medicament.", ["paracétomol"]),
        (
            "Le patient a un carcinome\nhépatacellulaire !",
            ["carcinome\nhépatacellulaire"],
        ),
    )

    for text, ents in texts:
        doc = blank_nlp(text)
        matches = list(matcher(doc))
        assert len(matches) > 0

        assert sorted([m.text for m in matcher(doc[2:], as_spans=True)]) == ents
        assert sorted([doc[s:e].text for _, s, e in matcher(doc[2:])]) == ents

        assert sorted([m.text for m in matcher(doc, as_spans=True)]) == ents
        assert sorted([doc[s:e].text for _, s, e in matcher(doc)]) == ents


@mark.parametrize("measure", ["dice", "cosine", "jaccard", "overlap"])
def test_without_normalizer(blank_nlp, measure):
    pattern = "matching"
    matcher = SimstringMatcher(
        blank_nlp.vocab, attr="NORM", threshold=0.6, measure=measure
    )

    matcher.build_patterns(
        blank_nlp,
        {
            "test": [pattern],
            "C220": ["carcinome hépatocellulaire", "carc. hépatocellulaire"],
            "N02BE01": ["paracétamol"],
        },
    )

    texts = (
        ("Ceci est un test de matching", ["matching"]),
        ("Ceci est un test de matchings", ["matchings"]),
        ("On prescrit du paracétomol, un médicament.", ["paracétomol"]),
        (
            "Le patient a un carcinome hépatacellulaire !",
            ["carcinome hépatacellulaire"],
        ),
    )

    for text, ents in texts:
        doc = blank_nlp(text)
        matches = list(matcher(doc))
        assert len(matches) > 0

        assert sorted([m.text for m in matcher(doc[2:], as_spans=True)]) == ents
        assert sorted([doc[s:e].text for _, s, e in matcher(doc[2:])]) == ents

        assert sorted([m.text for m in matcher(doc, as_spans=True)]) == ents
        assert sorted([doc[s:e].text for _, s, e in matcher(doc)]) == ents
