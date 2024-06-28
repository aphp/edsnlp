def test_covid(blank_nlp):
    examples = [
        ("J'ai vu le patient à cause d'une TS médicamenetuse.", "TS"),
        ("J'ai vu le patient à cause d'une ts médicamenetuse.", ""),
        ("J'ai vu le patient à cause d'une IMV.", "IMV"),
        ("surface TS", ""),
    ]

    blank_nlp.add_pipe("eds.suicide_attempt")

    for example, text in examples:
        doc = blank_nlp(example)

        if len(doc.ents) > 0:
            ent = doc.ents[0]
            assert ent.text == text
        else:
            assert text == ""
