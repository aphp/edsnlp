from edsnlp.connectors.labeltool import docs2labeltool

texts = [
    "Le patient est malade",
    "Le patient n'est pas malade",
    "Le patient est peut-être malade",
    "Le patient dit qu'il est malade",
]


def test_docs2labeltool(nlp):

    modifiers = ["negated", "hypothesis", "reported_speech"]

    docs = list(nlp.pipe(texts))

    df = docs2labeltool(docs, extensions=modifiers)
    assert len(df)

    df = docs2labeltool(docs)
    assert len(df)
