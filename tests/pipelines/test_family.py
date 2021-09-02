from typing import List
from edsnlp.utils.examples import parse_example

family_examples: List[str] = [
    "Le père du patient a eu un <ent family_=FAMILY>cancer du colon</ent>.",
    "Antécédents familiaux : <ent family_=FAMILY>diabète</ent>.",
    "Un <ent family_=PATIENT>relevé</ent> sanguin a été effectué.",
]


def test_family(nlp):

    for example in family_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for ent in entities:

            span = doc.char_span(ent.start_char, ent.end_char)

            for modifier in ent.modifiers:

                assert all(
                    [getattr(token._, modifier.key) == modifier.value for token in span]
                ), f"{modifier.key} labels don't match."
