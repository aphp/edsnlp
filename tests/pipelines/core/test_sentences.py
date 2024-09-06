from pytest import mark

import edsnlp

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


def test_newlines_double():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.sentences",
        config={
            "punct_chars": [],
            "ignore_excluded": False,
            "check_capitalized": False,
            "min_newline_count": 2,
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
