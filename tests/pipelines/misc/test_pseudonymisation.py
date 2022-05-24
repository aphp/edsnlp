from typing import List

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest import fixture

from edsnlp.pipelines.misc.pseudonymisation.date_patterns import pseudo_date_pattern
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    (
        "Le patient habite à <ent label=VILLE>Clermont-Ferrand</ent>, "
        "<ent label=ZIP>63 000</ent>"
    ),
    (
        "Le patient téléphone au <ent label=TEL>06 12 34 56 79</ent>, "
        "<ent label=TEL>0 6 1 2 3 4 5 6 7 9</ent>"
    ),
    (
        "<ent label=MAIL>medecin@aphp.fr</ent>, "
        "<ent label=MAIL>patient@test.example.com</ent>, "
        "<ent label=MAIL>patient @ test . example . com</ent>"
    ),
    "Date d'examen <ent label=date>0 6 0 9 2 0 2 2</ent>",
]


@fixture(scope="function")
def pseudo(blank_nlp):
    blank_nlp.add_pipe("eds.pseudonymisation")
    blank_nlp.add_pipe(
        "eds.dates",
        config=dict(
            absolute=pseudo_date_pattern,
            false_positive=[],
            as_ents=True,
        ),
    )
    return blank_nlp
    # return Pseudonymisation(blank_nlp, "NORM")


def test_pseudonymisation(pseudo):

    for example in examples:
        text, entities = parse_example(example=example)
        doc = pseudo(text)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]


@given(email=st.emails())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_email_detection(blank_nlp, pseudo, email):

    doc = blank_nlp(email)
    doc = pseudo(doc)

    assert len(doc.ents) == 1

    ent = doc.ents[0]

    assert len(email) == len(ent.text)

    assert ent.label_ == "MAIL"
