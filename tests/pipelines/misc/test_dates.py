from datetime import datetime

import pytz
import spacy
from pytest import fixture
from spacy.language import Language

from edsnlp.pipelines.misc.dates.models import AbsoluteDate, Direction, Mode
from edsnlp.utils.examples import parse_example

TZ = pytz.timezone("Europe/Paris")

examples = [
    "Le patient est venu en <ent year=2019>2019</ent> pour une consultation",
    "Le patient est venu <ent direction=PAST day=1>hier</ent>",
    "le <ent day=4 month=9 year=2021>04/09/2021</ent>",
    "Il est cas contact <ent direction=PAST week=1>depuis la semaine dernière</ent>",
    "le <ent day=9 month=8>09/08</ent>",
    "Le patient est venu le <ent day=4 month=8>4 août</ent>",
    "Le patient est venu le <ent day=4 month=8 hour=11 minute=13>4 août à 11h13</ent>",
    "Il est venu le <ent day=1 month=9>1er Septembre</ent> pour",
    "Il est venu en <ent month=10 year=2020>octobre 2020</ent> pour...",
    "Il est venu <ent direction=PAST month=3>il y a trois mois</ent> pour...",
    "Il lui était arrivé la même chose <ent direction=PAST year=1>il y a un an</ent>.",
    "Il est venu le <ent day=20 month=9 year=2001>20/09/2001</ent> pour...",
    "Consultation du <ent mode=FROM day=3 month=7 year=2019>03 07 19</ent>",
    "En <ent month=11 year=2017>11/2017</ent> stabilité sur...",
    "<ent direction=PAST month=3>depuis 3 mois</ent>",
    "- <ent month=12 year=2004>Décembre 2004</ent> :",
    "- <ent month=6 year=2005>Juin 2005</ent>:  ",
    # "-<ent month=6 year=2005>Juin 2005</ent>:  ",  # issues with "fr" language
    "<ent month=9 year=2017>sept 2017</ent> :",
    (
        "<ent direction=PAST year=1>il y a 1 an</ent> "
        "<ent mode=DURATION month=1>pdt 1 mois</ent>"
    ),
    (
        "Prélevé le : <ent day=22 month=4 year=2016>22/04/2016</ent> "
        "\n78 rue du Général Leclerc"
    ),
    "Le <ent day=7 month=1>07/01</ent>.",
    # "il est venu <ent year=0 direction=CURRENT>cette année</ent>",
    # "je vous écris <ent direction=CURRENT day=0>ce jour</ent> à propos du patient",
    "Il est venu en <ent month=8>août</ent>.",
]


@fixture(autouse=True)
def add_date_pipeline(blank_nlp: Language):
    blank_nlp.add_pipe("eds.dates", config=dict(detect_periods=True))


def test_dates_component(blank_nlp: Language):
    note_datetime = datetime(year=1993, month=9, day=23)

    for example in examples:
        text, entities = parse_example(example)

        doc = blank_nlp(text)

        assert len(doc.spans["dates"]) == len(entities)

        for span, entity in zip(doc.spans["dates"], entities):
            assert span.text == text[entity.start_char : entity.end_char]

            date = span._.date
            d = {modifier.key: modifier.value for modifier in entity.modifiers}
            if "direction" in d:
                d["direction"] = Direction[d["direction"]]
            if "mode" in d:
                d["mode"] = Mode[d["mode"]]

            assert date.dict(exclude_none=True) == d

            set_d = set(d)

            if isinstance(date, AbsoluteDate) and {"year", "month", "day"}.issubset(
                set_d
            ):
                d.pop("direction", None)
                d.pop("mode", None)
                assert date.to_datetime() == TZ.localize(datetime(**d))

            elif isinstance(date, AbsoluteDate):
                assert date.to_datetime() is None

                # no year
                if {"month", "day"}.issubset(set_d) and {"year"}.isdisjoint(set_d):
                    d["year"] = note_datetime.year
                    assert date.to_datetime(
                        note_datetime=note_datetime, **dict(enhance=True)
                    ) == TZ.localize(datetime(**d))

                # no day
                if {"month", "year"}.issubset(set_d) and {"day"}.isdisjoint(set_d):
                    d["day"] = 1
                    assert date.to_datetime(
                        note_datetime=note_datetime, **dict(enhance=True)
                    ) == TZ.localize(datetime(**d))

                # year only
                if {"year"}.issubset(set_d) and {"day", "month"}.isdisjoint(set_d):
                    d["day"] = 1
                    d["month"] = 1
                    assert date.to_datetime(
                        note_datetime=note_datetime, **dict(enhance=True)
                    ) == TZ.localize(datetime(**d))

                # month only
                if {"month"}.issubset(set_d) and {"day", "year"}.isdisjoint(set_d):
                    d["day"] = 1
                    d["year"] = note_datetime.year
                    assert date.to_datetime(
                        note_datetime=note_datetime, **dict(enhance=True)
                    ) == TZ.localize(datetime(**d))

            else:
                assert date.to_datetime()


def test_periods(blank_nlp: Language):

    period_examples = [
        "à partir de <ent>juin 2017 pendant trois semaines</ent>",
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
        "Il est cas contact <ent>depuis la semaine dernière</ent>, "
        "le <ent>09/08</ent> (<ent>2021-08-09</ent>)."
    )

    nlp = spacy.blank("eds")

    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.matcher", config=dict(terms={"contact": "contact"}))
    nlp.add_pipe("eds.dates", config=dict(on_ents_only=True))

    text, entities = parse_example(text)

    doc = nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["dates"]) == len(entities)

    for span, entity in zip(doc.spans["dates"], entities):
        assert span.text == text[entity.start_char : entity.end_char]
