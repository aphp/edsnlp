from typing import List
from edsnlp.utils.examples import parse_example

hypothesis_examples: List[str] = [
    "Plusieurs <ent hypothesis_=HYP>diagnostics</ent> sont envisagés.",
    "Suspicion de <ent hypothesis_=HYP>diabète</ent>.",
    "Le ligament est <ent hypothesis_=CERT>rompu</ent>.",
]


def test_hypothesis(nlp):

    for example in hypothesis_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for ent in entities:

            span = doc.char_span(ent.start_char, ent.end_char)

            for modifier in ent.modifiers:

                assert all(
                    [getattr(token._, modifier.key) == modifier.value for token in span]
                ), f"{modifier.key} labels don't match."
