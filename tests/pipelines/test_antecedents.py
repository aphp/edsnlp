from typing import List
from edsnlp.utils.examples import parse_example

antecedent_examples: List[str] = [
    "Antécédents d'<ent antecedent_=ATCD>AVC</ent>.",
    "atcd <ent antecedent_=ATCD>chirurgicaux</ent> : aucun.",
    "Le patient est <ent antecedent_=CURRENT>fumeur</ent>.",
    # Les sections ne sont pas utilisées par défaut
    "\nv Antecedents :\n- <ent antecedent_=CURRENT>appendicite</ent>\nv Motif :\n<ent antecedent_=CURRENT>malaise</ent>",
]


def test_antecedent(nlp):

    for example in antecedent_examples:

        text, entities = parse_example(example=example)

        doc = nlp(text)

        for ent in entities:

            span = doc.char_span(ent.start_char, ent.end_char)

            for modifier in ent.modifiers:

                assert all(
                    [getattr(token._, modifier.key) == modifier.value for token in span]
                ), f"{modifier.key} labels don't match."
