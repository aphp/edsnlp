from typing import List
from edsnlp.utils.examples import parse_match, parse_example, find_matches

hypothesis_examples: List[str] = [
    "Plusieurs <ent hypothesis_=HYP>diagnostics</ent> sont envisagés.",
    "Suspicion de <ent hypothesis_=HYP>diabète</ent>.",
    "Le ligament est <ent hypothesis_=CERT>rompu</ent>.",
]


def test_hypothesis(nlp):

    for example in hypothesis_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for entity in entities:

            span = doc.char_span(entity.start_char, entity.end_char)

            m1 = entity.modifiers[0]

            assert all(
                [token._.hypothesis_ == m1.value for token in span]
            ), f"{text} : Hypothesis labels don't match."
