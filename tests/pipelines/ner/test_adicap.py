from edsnlp.utils.examples import parse_example

d1v1 = "ABCD"
d1v2 = "BFEZ"
d2v1 = "0123"
d2v2 = "A1B3"
d2v3 = "ABC3"
d2v4 = "A990"
d2v5 = "A9AZ"
d2v6 = "0A12"

examples = [
    f"""1. Codification ADICAP : <ent text={d1v1 + d2v1}>{d1v1 + d2v1}</ent>.
    Une autre chose""",
    rf"""2. Codification ADICAP : <ent text={d1v1 + d2v2}>{d1v1 + d2v2}</ent>,\s
    <ent text={d1v1 + d2v3}>{d1v1 + d2v3}</ent>. Une autre chose""",
    f"""3. adicap : <ent text={d1v2 + d2v3}>{d1v2 + d2v3}</ent>,
    <ent text={d1v1 + d2v4}>{d1v1 + d2v4}</ent>. Une autre chose""",
    f"""4. Codification  : <ent text={d1v1 + d2v6}>{d1v1 + d2v6}</ent>.
    J'aime edsnlp. : {d1v2 + d2v3}.  Une autre chose""",
    f"""5. J'aime edsnlp. : {d1v2 + d2v5}.  Une autre chose""",
]


def test_scores(blank_nlp):
    if blank_nlp.lang == "eds":
        blank_nlp.add_pipe("eds.adicap")

        for example in examples:

            text, expected_entities = parse_example(example=example)

            doc = blank_nlp(text)

            for expected, ent in zip(expected_entities, doc.ents):
                assert text[expected.start_char : expected.end_char] == ent.text
                assert expected.modifiers[0].value == ent._.adicap.code
                assert len(ent._.adicap.dict()) > 0


def test_local_spelling(blank_nlp):
    blank_nlp.add_pipe("eds.adicap")
    txt = "Codification ADICAP : B.H.HP.A7A0 . Autre chose"
    assert blank_nlp(txt).ents[0]._.adicap.dict() == {
        "behaviour_type": "CANCER INVASIF",
        "code": "BHHPA7A0",
        "organ": "PROSTATE",
        "pathology": "PATHOLOGIE TUMORALE",
        "pathology_type": "ADENOCARCINOME INVASIF (SAI)",
        "sampling_mode": "BIOPSIE CHIRURGICALE",
        "technic": "HISTOLOGIE ET CYTOLOGIE PAR INCLUSION",
    }
