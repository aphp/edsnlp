def test_covid(blank_nlp):
    examples = [
        ("Patient admis pour coronavirus", "coronavirus"),
        ("Patient admis pour pneumopathie à coronavirus", "pneumopathie à coronavirus"),
    ]

    blank_nlp.add_pipe("eds.covid")

    for example, text in examples:
        doc = blank_nlp(example)

        covid = doc.ents[0]
        assert covid.text == text
