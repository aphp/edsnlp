from datetime import datetime

import pytest
import pytz
import spacy
from pytest import fixture
from spacy.language import Language

from edsnlp.pipelines.misc.dates.models import AbsoluteDate, Relative
from edsnlp.utils.examples import parse_example

TZ = pytz.timezone("Europe/Paris")

examples = [
    (
        "Le patient est venu en <ent norm='2019-??-??' year=2019>2019</ent> pour une "
        "consultation"
    ),
    "Le patient est venu <ent norm='-1 day' direction=past day=1>hier</ent>",
    "le <ent norm='2021-09-04' day=4 month=9 year=2021>04/09/2021</ent>",
    (
        "Il est cas contact <ent norm='-1 week' direction=past week=1>"
        "depuis la semaine dernière</ent>"
    ),
    "le <ent norm='????-08-09' day=9 month=8>09/08</ent>",
    "Le patient est venu le <ent norm='????-08-04' day=4 month=8>4 août</ent>",
    (
        "Le patient est venu le <ent norm='????-08-04 11h13m' day=4 month=8 "
        "hour=11 minute=13>4 août à 11h13</ent>"
    ),
    "Il est venu le <ent norm='????-09-01' day=1 month=9>1er Septembre</ent> pour",
    (
        "Il est venu en <ent norm='2020-10-??' month=10 year=2020>octobre 2020</ent> "
        "pour..."
    ),
    (
        "Il est venu <ent norm='-3 months' direction=past month=3>il y a "
        "trois mois</ent> pour..."
    ),
    (
        "Il lui était arrivé la même chose <ent norm='-1 year' "
        "direction=past year=1>il y a un an</ent>."
    ),
    (
        "Il est venu le <ent norm='2001-09-20' day=20 month=9 "
        "year=2001>20/09/2001</ent> pour..."
    ),
    (
        "Consultation du <ent norm='2019-07-03' bound=from "
        "day=3 month=7 year=2019>03 07 19</ent>"
    ),
    "En <ent norm='2017-11-??' month=11 year=2017>11/2017</ent> stabilité sur...",
    "<ent norm='-3 months' direction=past month=3>depuis 3 mois</ent>",
    "- <ent norm='2004-12-??' month=12 year=2004>Décembre 2004</ent> :",
    "- <ent norm='2005-06-??' month=6 year=2005>Juin 2005</ent>:  ",
    # "-<ent norm=" month=6 year=2005>Juin 2005</ent>:  ",  # issues with "fr" language
    "<ent norm='2017-09-??' month=9 year=2017>sept 2017</ent> :",
    (
        "<ent norm='-1 year' direction=past year=1>il y a 1 an</ent> "
        "<ent norm='during 1 month' mode=duration month=1>pdt 1 mois</ent>"
    ),
    (
        "Prélevé le : <ent norm='2016-04-22' day=22 month=4 year=2016>22/04/2016</ent> "
        "\n78 rue du Général Leclerc"
    ),
    "Le <ent norm='????-01-07' day=7 month=1>07/01</ent>.",
    "Il est venu en <ent norm='????-08-??' month=8>août</ent>.",
    "Il est venu <ent norm='~0 day' day=0 direction=current>ce jour</ent>.",
    "CS le <ent norm='2017-01-11' day=11 month=1 year=2017>11-01-2017</ent> 1/3",
    "Vu le <ent norm='2017-01-11' day=11 month=1 year=2017>11 janvier\n2017</ent> .",
]


@fixture(autouse=True)
def add_date_pipeline(blank_nlp: Language):
    blank_nlp.add_pipe("eds.dates", config=dict(detect_periods=True, as_ents=True))


def test_dates_component(blank_nlp: Language):
    note_datetime = datetime(year=1993, month=9, day=23)

    for example in examples:
        text, entities = parse_example(example)

        doc = blank_nlp(text)
        spans = sorted(doc.spans["dates"] + doc.spans["durations"])

        assert len(spans) == len(entities)
        assert len(doc.ents) == len(entities)

        for span, entity in zip(spans, entities):
            assert span.text == text[entity.start_char : entity.end_char]

            date = span._.date if span.label_ == "date" else span._.duration
            d = {modifier.key: modifier.value for modifier in entity.modifiers}
            norm = d.pop("norm")
            if "direction" in d:
                d["mode"] = "relative"
            if "mode" not in d:
                d["mode"] = "absolute"

            assert date.dict(exclude_none=True) == d
            assert date.norm() == norm

            set_d = set(d)

            d.pop("mode", None)
            d.pop("direction", None)
            d.pop("bound", None)
            if isinstance(date, AbsoluteDate) and {"year", "month", "day"}.issubset(
                set_d
            ):
                assert date.to_datetime() == TZ.localize(datetime(**d))

            elif isinstance(date, AbsoluteDate):
                assert date.to_datetime() is None

                # no year
                if {"month", "day"}.issubset(set_d) and {"year"}.isdisjoint(set_d):
                    d["year"] = note_datetime.year
                    assert date.to_datetime(
                        note_datetime=note_datetime, infer_from_context=True
                    ) == TZ.localize(datetime(**d))

                # no day
                if {"month", "year"}.issubset(set_d) and {"day"}.isdisjoint(set_d):
                    d["day"] = 1
                    assert date.to_datetime(
                        note_datetime=note_datetime, infer_from_context=True
                    ) == TZ.localize(datetime(**d))

                # year only
                if {"year"}.issubset(set_d) and {"day", "month"}.isdisjoint(set_d):
                    d["day"] = 1
                    d["month"] = 1
                    assert date.to_datetime(
                        note_datetime=note_datetime, infer_from_context=True
                    ) == TZ.localize(datetime(**d))

                # month only
                if {"month"}.issubset(set_d) and {"day", "year"}.isdisjoint(set_d):
                    d["day"] = 1
                    d["year"] = note_datetime.year
                    assert date.to_datetime(
                        note_datetime=note_datetime, infer_from_context=True
                    ) == TZ.localize(datetime(**d))

            elif isinstance(date, Relative):
                assert date.to_datetime() is None
            else:
                assert date.to_duration()
                assert date.to_datetime(note_datetime=note_datetime)


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


@pytest.mark.parametrize("with_time", [False, True])
def test_time(with_time: bool):
    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.dates", config={"detect_time": with_time})

    if with_time:
        time_examples = [
            "Vu le <ent norm='2012-01-11 11h34m'>11/01/2012 à 11h34</ent> pour radio.",
        ]
    else:
        time_examples = [
            "Vu le <ent norm='2012-01-11'>11/01/2012</ent> à 11h34 pour radio.",
        ]

    for example in time_examples:
        text, entities = parse_example(example)

        doc = nlp(text)

        spans = sorted(doc.spans["dates"] + doc.spans["durations"])

        assert len(spans) == len(entities)

        for span, entity in zip(spans, entities):
            assert span.text == text[entity.start_char : entity.end_char]
            norm = next(m.value for m in entity.modifiers if m.key == "norm")
            assert span._.date.norm() == norm


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

        assert len((*doc.spans["dates"], *doc.spans["durations"])) == 0


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

    spans = sorted(doc.spans["dates"] + doc.spans["durations"])

    assert len(spans) == len(entities)

    for span, entity in zip(spans, entities):
        assert span.text == text[entity.start_char : entity.end_char]


def test_illegal_dates(blank_nlp):
    texts = (
        " Le 31/06/17, la dernière dose.",
        " Le 30/02/18 n'est pas une vraie date",
    )
    for text in texts:
        doc = blank_nlp(text)
        ent = sorted((*doc.spans["dates"], *doc.spans["durations"]))[0]
        assert ent._.date.to_datetime() is None
