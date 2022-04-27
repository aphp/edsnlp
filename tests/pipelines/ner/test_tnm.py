from edsnlp.utils.examples import parse_example

examples = [
    "TNM: <ent norm=aTxN1M0>aTxN1M0</ent>",
    "TNM: <ent norm=pTxN1M0>p Tx N1M 0</ent>",
    "TNM: <ent norm='pTxN1M0 (UICC 2020)'>p Tx N1M 0 (UICC 20)</ent>",
    "TNM: <ent norm='aTxN1M0 (UICC 1968)'>aTxN1M0 (UICC 68)</ent>",
]


def test_scores(blank_nlp):

    blank_nlp.add_pipe("eds.TNM")

    for example in examples:

        text, entities = parse_example(example=example)

        doc = blank_nlp(text)

        for entity, ent in zip(entities, doc.ents):
            norm = entity.modifiers[0].value
            assert norm == ent._.value.norm()
