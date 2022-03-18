from typing import Dict, List, Tuple

from pytest import mark

from edsnlp.pipelines.misc.context import ContextPseudonymisation
from edsnlp.utils.examples import parse_example

examples: List[Tuple[str, Dict[str, List[str]]]] = [
    (
        "Le patient s'appelle <ent>Jean</ent> <ent>Dupont</ent>",
        dict(NOM=["Dupont"], PRENOM=["Jean"]),
    ),
]


@mark.parametrize("phrase", [True, False])
def test_context_pseudonymisation(blank_nlp, phrase):
    pseudo = ContextPseudonymisation(blank_nlp, attr="NORM", phrase=phrase)

    for example, context in examples:
        text, entities = parse_example(example=example)
        doc = blank_nlp(text)
        doc._context = context
        doc = pseudo(doc)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]
