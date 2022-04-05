import spacy
from pytest import fixture
from spacy.language import Language

from edsnlp.utils.examples import parse_example

examples = [
    "Le patient est venu <ent direction=ago day=1>hier</ent>",
    "le <ent day=4 month=9 year=2021>04/09/2021</ent>",
    "Il est cas contact depuis <ent direction=ago week=1>la semaine dernière</ent>",
    # "le <ent day=9 month=8>9/8</ent>",
    "le <ent day=9 month=8>09/08</ent>",
    "Le patient est venu le <ent day=4 month=8>4 août</ent>",
    "Le patient est venu le <ent day=4 month=8 hour=11 minute=13>4 août à 11h13</ent>",
    # "Le patient est venu en <ent year=2019>2019</ent> pour une consultation",
    "Il est venu le <ent day=1 month=9>1er Septembre</ent> pour",
    "Il est venu en <ent month=10 year=2020>octobre 2020</ent> pour...",
    "Il est venu <ent direction=ago month=3>il y a trois mois</ent> pour...",
    "Il lui était arrivé la même chose <ent direction=ago year=1>il y a un an</ent>.",
    "Il est venu le <ent day=20 month=9 year=2001>20/09/2001</ent> pour...",
    "Consultation du <ent direction=since day=3 month=7 year=19>03 07 19</ent>",
    "En <ent month=11 year=2017>11/2017</ent> stabilité sur...",
    "<ent direction=since month=3>depuis 3 mois</ent>",
    "- <ent month=12 year=2004>Décembre 2004</ent> :",
    "- <ent month=6 year=2005>Juin 2005</ent>:  ",
    # "-<ent month=6 year=2005>Juin 2005</ent>:  ",  # issues with "fr" language
    "<ent month=9 year=2017>sept 2017</ent> :",
    (
        "<ent direction=ago year=1>il y a 1 an</ent> "
        "<ent direction=for month=1>pdt 1 mois</ent>"
    ),
    (
        "Prélevé le : <ent day=22 month=4 year=2016>22/04/2016</ent> "
        "\n78 rue du Général Leclerc"
    ),
    "Le <ent day=7 month=1>07/01</ent>.",
    # "il est venu <ent day=4 month=8>cette année</ent>",
    # "je vous écris <ent direction= day=1>ce jour</ent> à propos du patient",
]


@fixture(autouse=True)
def add_date_pipeline(blank_nlp: Language):
    blank_nlp.add_pipe("eds.dates")


def test_dates_component(blank_nlp: Language):

    for example in examples:
        text, entities = parse_example(example)

        doc = blank_nlp(text)

        assert len(doc.spans["dates"]) == len(entities)

        for span, entity in zip(doc.spans["dates"], entities):
            assert span.text == text[entity.start_char : entity.end_char]

            date = {modifier.key: modifier.value for modifier in entity.modifiers}

            assert span._.date.dict(exclude_none=True) == date


def test_periods(blank_nlp: Language):

    period_examples = [
        "en <ent>juin 2017 pendant trois semaines</ent>",
        "du <ent>5 juin au 6 juillet</ent>",
    ]

    for example in period_examples:
        text, entities = parse_example(example)

        doc = blank_nlp(text)

        assert len(doc.spans["periods"]) == len(entities)

        for span, entity in zip(doc.spans["periods"], entities):
            assert span.text == text[entity.start_char : entity.end_char]


def test_false_positives(blank_nlp: Language):

    counter_examples = [
        "page 1/1",  # Often found in the form `1/1` only
        "40 00",
        "06 12 34 56 78",
        "bien mais",
        "thierry",
        "436",
        "12.0-16",
        "27.0-33",
        "7.0-11",
        "03-0.70",
        "4.09-11",
        "2/2CR Urgences PSL",
        "Dextro : 5.7 mmol/l",
        "2.5",
    ]

    for example in counter_examples:
        doc = blank_nlp(example)

        assert len(doc.spans["dates"]) == 0


def test_dates_on_ents_only():

    text = (
        "Le patient est venu hier (le 04/09/2021) pour un test PCR.\n"
        "Il est cas contact depuis la semaine dernière, le 09/08 (2021-08-09)."
    )

    nlp = spacy.blank("eds")

    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.matcher", config=dict(terms={"contact": "contact"}))
    nlp.add_pipe("eds.dates", config=dict(on_ents_only=True))

    doc = nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["dates"]) == 3
