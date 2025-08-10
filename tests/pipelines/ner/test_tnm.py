from edsnlp.pipes.ner.tnm.patterns import tnm_pattern
from edsnlp.utils.examples import parse_example

examples = [
    "TNM: <ent norm=aTxN1M0>aTxN1M0</ent>",
    "TNM: <ent norm=pTxN1M0>p Tx N1M 0</ent>",
    "TNM: <ent norm='pTxN1M0 (UICC 2020)'>p Tx N1M 0 (UICC 20)</ent>",
    "TNM: <ent norm='aTxN1M0 (UICC 1968)'>aTxN1M0 (UICC 68)</ent>",
    "TNM: <ent norm=aTxN1R2>aTxN1 R2</ent>",
    "TNM: <ent norm='pT2cN0R0 (TNM 2010)'>pT2c N0 R0 (TNM 2010)</ent>",
    "TNM: <ent norm=aTxN1M0>aTx / N1 / M0</ent>",
    "TNM: <ent norm=pT2N1mi>pT2 N1mi</ent>",
    "TNM: <ent norm=pT1mN1M0>pT1(m)N1 M0</ent>",
    "TNM: <ent norm=pT1bN0sn>pT1bN0(sn)</ent>",
    "TNM: <ent norm=pT1N1M0>pT1 pN1 M0</ent>\n \n ",
    "TNM: <ent norm=aTxN1M0>aTxN1M0</ent> ",
    "TNM: <ent norm=cT3N0M0>cT3N0M0</ent> \n \n",
    "TNM: PT",
    "TNM: p    T \n",
    "TNM: a T \n",
    "TNM: <ent norm=p>pT</ent> \n  \n0",
]


def test_scores(blank_nlp):
    blank_nlp.add_pipe("eds.tnm", config=dict(pattern=tnm_pattern))

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)

        assert len(entities) == len(doc.ents)

        for entity, ent in zip(entities, doc.ents):
            norm = entity.modifiers[0].value
            assert ent.text == text[entity.start_char : entity.end_char]
            assert norm == ent._.value.norm()
