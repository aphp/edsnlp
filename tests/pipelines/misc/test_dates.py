from datetime import date, datetime, timedelta

from dateparser import DateDataParser
from pytest import fixture, raises
from spacy.language import Language
from spacy.tokens import Span

from edsnlp.pipelines.misc.dates import Dates
from edsnlp.pipelines.misc.dates.dates import apply_groupdict, date_parser
from edsnlp.pipelines.misc.dates.factory import DEFAULT_CONFIG
from edsnlp.pipelines.misc.dates.parsing import day2int_fast, month2int_fast


@fixture(scope="session")
def parser():
    return date_parser


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
        assert parser(test).date() == answer


def test_incomplete_dates(parser: DateDataParser):
    tests = [
        ("en mars 2010", date(2010, 3, 1)),
        ("en 2019", date(2019, 1, 1)),
    ]

    for test, answer in tests:
        assert parser(test).date() == answer

    no_year_date = parser("le 3 juillet").date()
    assert no_year_date.month == 7
    assert no_year_date.day == 3


def test_parser_relative(parser: DateDataParser):
    tests = [
        ("hier", [timedelta(days=-1)]),
        (
            "le mois dernier",
            [
                timedelta(days=-31),
                timedelta(days=-30),
                timedelta(days=-29),
                timedelta(days=-28),
            ],
        ),
        ("il y a trois jours", [timedelta(days=-3)]),
        ("l'année dernière", [timedelta(days=-365), timedelta(days=-366)]),
        # ("l'an dernier", timedelta(days=-365)),
    ]

    for test, answers in tests:
        assert any([parser(test).date() == (date.today() + a) for a in answers])


text = (
    "Le patient est venu hier (le 04/09/2021) pour un test PCR.\n"
    "Il est cas contact depuis la semaine dernière, le 09/08 (2021-08-09)."
)


@fixture
def dates(blank_nlp: Language):
    return Dates(
        blank_nlp,
        **DEFAULT_CONFIG,
    )


def test_dateparser_failure_cases(
    blank_nlp: Language, dates: Dates, parser: DateDataParser
):
    examples = [
        "le premier juillet 2021",
        "l'an dernier",
    ]

    for example in examples:
        assert parser(example) is None

        doc = blank_nlp(example)
        doc = dates(doc)

        d = doc.spans["dates"][0]

        assert d._.date == "????-??-??"


def test_dates_on_ents_only(blank_nlp: Language):

    blank_nlp.add_pipe("matcher", config=dict(terms={"contact": "contact"}))
    blank_nlp.add_pipe("dates", config=dict(on_ents_only=True))

    doc = blank_nlp(text)

    assert len(doc.ents) == 1

    assert len(doc.spans["dates"]) == 3


def test_dates_component(blank_nlp: Language, dates: Dates):

    doc = blank_nlp(text)

    with raises(KeyError):
        doc.spans["dates"]

    doc = dates(doc)

    d1, d2, d3, d4, d5 = doc.spans["dates"]

    assert d1._.date == "TD-1"
    assert d2._.date == "2021-09-04"
    assert d3._.date == "TD-7"
    assert d4._.date == "????-08-09"
    assert d5._.date == "2021-08-09"


def test_dates_with_base_date(blank_nlp: Language, dates: Dates):

    doc = blank_nlp(text)
    doc = dates(doc)

    doc._.note_datetime = datetime(2020, 10, 10)

    d1, d2, d3, d4, d5 = doc.spans["dates"]

    assert d1._.date == "2020-10-09"
    assert d2._.date == "2021-09-04"
    assert d3._.date == "2020-10-03"
    assert d4._.date == "2020-08-09"
    assert d5._.date == "2021-08-09"


def test_absolute_dates_patterns(blank_nlp: Language, dates: Dates):

    examples = [
        ("Objet : Consultation du 03 07 19", "2019-07-03"),
        ("Objet : Consultation du 03 juillet 19", "2019-07-03"),
        ("Objet : Consultation du 3 juillet 19", "2019-07-03"),
        ("Objet : Consultation du 03-07-19", "2019-07-03"),
        ("Objet : Consultation du 03-07-1993", "1993-07-03"),
        ("Objet : Consultation du 03.07.1993", "1993-07-03"),
        ("Objet : Consultation du 1993-12-02", "1993-12-02"),
        ("Objet : Consultation du 1993.12.02", "1993-12-02"),
        ("en 09/17", "2017-09-01"),
        ("13/07/2021 13:21", "2021-07-13"),
        ("Objet : Compte-Rendu de Consultation du 03/10/2018", "2018-10-03"),
        ("Objet : Compte-Rendu de Consultation du 03/10/2019 à 15h.", "2019-10-03"),
    ]

    for example, answer in examples:
        doc = blank_nlp(example)
        doc = dates(doc)

        date = doc.spans["dates"][0]

        assert date._.date == answer


def test_patterns(blank_nlp: Language, dates: Dates):

    examples = [
        "Le patient est venu le 4 août",
        "Le patient est venu le 4 août à 11h13",
        "Le patient est venu en 2019 pour une consultation",
        "Le patient est venu le 1er septembre pour une consultation",
        "Le patient est venu le 1er Septembre pour une consultation",
        "Le patient est venu en octobre 2020 pour une consultation",
        "Le patient est venu il y a trois mois pour une consultation",
        "Le patient est venu il y a un an pour une consultation",
        "Il lui était arrivé la même chose il y a un an.",
        "Le patient est venu le 20/09/2001 pour une consultation",
        "Objet : Consultation du 03 07 19",
        "En 11/2017 stabilité sur l'IRM médullaire des lésions",
        "depuis 3 mois",
        "- Décembre 2004 :",
        "- Juin 2005:  ",
        "-Avril 2011 :",
        "sept 2017 :",
        # "il y a 1 an pdt 1 mois",
        "Prélevé le : 22/04/2016 \n78 rue du Général Leclerc",
        "Le 07/01.",
        "il est venu cette année",
        "je vous écris ce jour à propos du patient",
    ]

    for example in examples:
        doc = blank_nlp(example)
        doc = dates(doc)

        assert len(doc.spans["dates"]) == 1, doc.spans["dates"]


def test_false_positives(blank_nlp: Language, dates: Dates):

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
        doc = dates(doc)

        assert len(doc.spans["dates"]) == 0


def test_date_process(blank_nlp: Language, dates: Dates):

    examples = [
        ("2019-11-21", ["full_date", "absolute"]),
        ("22/10/2019", ["absolute"]),
        ("04/11/2019", ["absolute"]),
        ("22/10", ["no_year"]),
        ("10/19", ["no_day"]),
        ("10/11", ["no_year", "no_day"]),
    ]

    for example, labels in examples:
        doc = blank_nlp(example)
        ds = list(dates.regex_matcher(doc, as_spans=True))

        assert [date.label_ for date in ds] == labels


def test_number_of_instances(blank_nlp):
    blank_nlp.add_pipe("dates")

    examples = [
        (
            (
                "COMPTE RENDU D'HOSPITALISATION du 22/10/2019 au 05/11/2019\n"
                "MOTIF D'HOSPITALISATION\n"
                "Madame XX XX XX, née le 15/09/1973, "
                "âgée de 46 ans, a été hospitalisée du 22/10/2019\n"
                "au 04/11/2019 pour ischémie subaiguë gauche sur thrombose "
                "d'un pontage ilio-femoral profond."
            ),
            5,
        )
    ]

    for example, n in examples:
        doc = blank_nlp(example)
        assert len(doc.spans["dates"]) == n


# def test_dates_with_time(blank_nlp):
#     blank_nlp.add_pipe("dates")

#     examples = [
#         ("le trois septembre à 8h", "trois septembre à 8h"),
#         ("22/10/2019 09:12", "22/10/2019 09:12"),
#     ]

#     for example, text in examples:
#         doc = blank_nlp(example)
#         d = doc.spans["dates"][0]
#         assert d.text == text


def test_parse_day_fast():
    tests = [
        ("5", 5),
        ("trois", 3),
        ("1", 1),
        ("premier", 1),
        ("dixsept", 17),
        ("dix-sept", 17),
        ("vingt-neuf", 29),
        ("trente et un", 31),
        ("trente-et-un", 31),
        ("troi", None),
        ("janvier", None),
    ]

    for test, answer in tests:
        parsed = day2int_fast(test)
        if answer is None:
            assert parsed is None
        else:
            assert parsed == answer, test


def test_parse_month_fast():
    tests = [
        ("janv", 1),
        ("février", 2),
        ("fevrier", 2),
        ("jan", None),
        ("neuf", None),
    ]

    for test, answer in tests:
        parsed = month2int_fast(test)
        if answer is None:
            assert parsed is None
        else:
            assert parsed == answer


def test_groupdict_parsing(blank_nlp, dates: Dates):

    if not Span.has_extension("groupdict"):
        Span.set_extension("groupdict", default=dict())

    examples = [
        ("Le 03/01/2012 à 9h15m32", dict(day=3, month=1, year=2012)),
        # ("Le 03/01 2012 à 9h15m32", dict(day=3, month=1, year=2012)),
        ("Le 3 janvier 2012 à 9h15m32", dict(day=3, month=1, year=2012)),
        ("Le trois janvier 2012 à 9h15m32", dict(day=3, month=1, year=2012)),
        ("Le trente et un décembre 2012 à 9h15m32", dict(day=31, month=12, year=2012)),
    ]

    for text, d in examples:
        doc = blank_nlp(text)
        spans = apply_groupdict(
            dates.regex_matcher(
                doc,
                as_spans=True,
                return_groupdict=True,
            )
        )
        span = list(spans)[0]

        assert span._.groupdict == d, text
