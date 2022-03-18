from typing import List

from edsnlp.pipelines.misc.pseudonymisation import Pseudonymisation
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    "Le patient habite à <ent label=VILLE>Clermont-Ferrand</ent>",
    "Le patient téléphone au <ent label=TEL>06 12 34 56 79</ent>",
]


def test_pseudonymisation(blank_nlp):
    pseudo = Pseudonymisation(blank_nlp, "NORM")

    for example in examples:
        text, entities = parse_example(example=example)
        doc = blank_nlp(text)
        doc = pseudo(doc)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]
