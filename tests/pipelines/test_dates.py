from dateparser import DateDataParser
from pytest import fixture, raises
from datetime import date, timedelta, datetime

from edsnlp.pipelines.dates import Dates, terms


@fixture(scope="session")
def parser():
    return DateDataParser(languages=["fr"])


def test_parser_absolute(parser: DateDataParser):
    tests = [
        ("le 3 juillet 2020", date(2020, 7, 3)),
        ("le 3/7/2020", date(2020, 7, 3)),
        ("le 03 07 20", date(2020, 7, 3)),
        ("03/07/2020", date(2020, 7, 3)),
        ("03.07.20", date(2020, 7, 3)),
        ("1er juillet 2021", date(2021, 7, 1)),
        # ("le premier juillet 2021", date(2021, 7, 1)),
    ]

    for test, answer in tests:
        assert parser.get_date_data(test).date_obj.date() == answer


def test_parser_relative(parser: DateDataParser):
    tests = [
        ("hier", timedelta(days=-1)),
        ("le mois dernier", timedelta(days=-31)),
        ("il y a trois jours", timedelta(days=-3)),
        ("l'année dernière", timedelta(days=-365)),
        # ("l'an dernier", timedelta(days=-365)),
    ]

    for test, answer in tests:
        assert parser.get_date_data(test).date_obj.date() == date.today() + answer


text = (
    "Le patient est venu hier (le 04/09/2021) pour un test PCR.\n"
    "Il est cas contact depuis la semaine dernière."
)


@fixture
def dates(nlp):
    return Dates(
        nlp,
        absolute=terms.absolute,
        relative=terms.relative,
    )


def test_dates_component(nlp, dates):

    doc = nlp(text)

    with raises(KeyError):
        doc.spans["dates"]

    doc = dates(doc)

    assert len(doc.spans["dates"]) == 3

    d1, _, d3 = doc.spans["dates"]

    assert d1.label_ == "TD-1"
    assert d3.label_ == "TD-7"


def test_dates_with_base_date(nlp, dates):

    doc = nlp(text)
    doc._.note_datetime = datetime(2020, 10, 10)
    doc = dates(doc)

    d1, _, d3 = doc.spans["dates"]

    assert d1.label_ == "2020-10-09"
    assert d3.label_ == "2020-10-03"


def test_patterns(nlp, dates):

    examples = [
        "Le patient est venu en 2019 pour une consultation",
        "Le patient est venu en octobre 2020 pour une consultation",
        "Le patient est venu il y a trois mois pour une consultation",
        "Le patient est venu le 20/09/2001 pour une consultation",
    ]

    for example in examples:
        doc = nlp(example)
        doc = dates(doc)

        assert len(doc.spans["dates"]) == 1
