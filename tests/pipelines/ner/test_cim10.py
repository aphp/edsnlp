from spacy.language import Language

from edsnlp.utils.examples import parse_example

examples = [
    "Patient admis pour <ent code=A01>fièvres typhoïde et paratyphoïde</ent>",
    "Patient admis pour <ent code=C221>C2.21</ent>",
]


def test_cim10(blank_nlp: Language):

    blank_nlp.add_pipe("eds.cim10")

    for text, entities in map(parse_example, examples):
        doc = blank_nlp(text)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]
            assert ent.kb_id_ == entity.modifiers[0].value
