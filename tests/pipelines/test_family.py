from typing import List
from edsnlp.utils.examples import parse_match, parse_example, find_matches

family_examples: List[str] = [
    "Le père du patient a eu un <ent family_=FAMILY>cancer du colon</ent>.",
    "Antécédents familiaux : <ent family_=FAMILY>diabète</ent>.",
    "Un <ent family_=PATIENT>relevé</ent> sanguin a été effectué.",
]


def test_family(nlp):

    for example in family_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for entity in entities:

            span = doc.char_span(entity.start_char, entity.end_char)

            m1 = entity.modifiers[0]

            assert all(
                [token._.family_ == m1.value for token in span]
            ), f"{text} : Family labels don't match."
