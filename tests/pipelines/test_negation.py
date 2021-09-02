from typing import List
from edsnlp.utils.examples import parse_example

negation_examples: List[str] = [
    "Le patient n'est pas <ent polarity_=NEG>malade</ent>.",
    "Aucun <ent polarity_=NEG>traitement</ent>.",
    "Le <ent polarity_=AFF>scan</ent> révèle une grosseur.",
]


def test_negation(nlp):

    for example in negation_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for ent in entities:

            span = doc.char_span(ent.start_char, ent.end_char)

            for modifier in ent.modifiers:

                assert all(
                    [getattr(token._, modifier.key) == modifier.value for token in span]
                ), f"{modifier.key} labels don't match."
