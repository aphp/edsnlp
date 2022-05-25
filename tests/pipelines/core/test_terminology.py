from spacy.language import Language

from edsnlp.utils.examples import parse_example

example = "1g de <ent kb_id=paracetamol>doliprane</ent>"


def test_terminology(blank_nlp: Language):
    blank_nlp.add_pipe(
        "eds.terminology",
        config=dict(
            label="drugs",
            terms=dict(paracetamol=["doliprane", "tylenol", "paracetamol"]),
            attr="NORM",
        ),
    )

    text, entities = parse_example(example)

    doc = blank_nlp(text)

    assert len(entities) == len(doc.ents)

    for ent, entity in zip(doc.ents, entities):
        assert ent.text == text[entity.start_char : entity.end_char]
        assert ent.kb_id_ == entity.modifiers[0].value
