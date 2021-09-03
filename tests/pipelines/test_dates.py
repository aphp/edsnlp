from dateparser import DateDataParser
from pytest import fixture
from datetime import date


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
    ]

    for test, answer in tests:
        assert parser.get_date_data(test).date_obj.date() == answer
