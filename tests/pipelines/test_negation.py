from typing import List
from edsnlp.utils.examples import parse_match, parse_example, find_matches

negation_examples: List[str] = [
    "Le patient n'est pas <ent polarity_=NEG>malade</ent>.",
    "Aucun <ent polarity_=NEG>traitement</ent>.",
    "Le <ent polarity_=AFF>scan</ent> révèle une grosseur.",
]


def test_negation(nlp):

    for example in negation_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for entity in entities:

            span = doc.char_span(entity.start_char, entity.end_char)

            m1 = entity.modifiers[0]

            assert all(
                [token._.polarity_ == m1.value for token in span]
            ), f"{text} : Polarity labels don't match. {[token._.polarity_ for token in span]}"
