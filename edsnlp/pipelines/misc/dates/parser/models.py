import re
from typing import Dict, Optional

from pydantic import BaseModel, root_validator

from edsnlp.pipelines.misc.dates.patterns.relative import specific_dict
from edsnlp.utils.regex import make_pattern

patterns = [
    r"month_(?P<month>\d\d)",
    r"day_(?P<day>\d\d)",
    r"number_(?P<number>\d\d)",
    r"direction_(?P<direction>\w+)",
    r"unit_(?P<unit>\w+)",
    r"specific_(?P<specific>.+)",
]

LETTER_PATTERN = re.compile(make_pattern(patterns))


class BaseDate(BaseModel):
    @root_validator(pre=True)
    def validate_strings(cls, d: Dict):
        result = d.copy()

        for k, v in d.items():
            if v is not None:
                match = LETTER_PATTERN.match(k)
                if match:
                    result.update({k: v for k, v in match.groupdict().items() if v})
        return result


class AbsoluteDate(BaseDate):

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None


class RelativeDate(BaseDate):

    direction: Optional[str] = None

    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    @root_validator(pre=True)
    def parse_unit(cls, d: Dict[str, str]):
        unit = d.get("unit")

        if unit:
            d[unit] = d.get("number")

        return d

    @root_validator(pre=True)
    def handle_specifics(cls, d: Dict[str, str]):

        specific = d.get("specific")
        specific = specific_dict.get(specific)

        if specific:
            d.update(specific)

        return d
