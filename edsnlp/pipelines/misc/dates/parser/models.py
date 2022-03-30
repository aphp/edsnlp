from typing import Optional

from pydantic import BaseModel, root_validator, validator

from .absolute import parse_day, parse_month
from .relative import parse_direction, parse_number, parse_specific


class AbsoluteDate(BaseModel):

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    @validator("month", pre=True)
    def validate_month(cls, v):
        return parse_month(v)

    @validator("day", pre=True)
    def validate_day(cls, v):
        return parse_day(v)


class RelativeDate(BaseModel):

    direction: Optional[str] = None

    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    @validator(
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        pre=True,
    )
    def validate_number(cls, v):
        return parse_number(v)

    @validator("direction", pre=True)
    def validate_direction(cls, v):
        return parse_direction(v)

    @root_validator(pre=True)
    def handle_specifics(cls, values):
        if "specific" in values:
            values.update(parse_specific(values["specific"]))
        return values
