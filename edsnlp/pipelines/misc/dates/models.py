from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union

import pendulum
from pydantic import BaseModel, root_validator, validator

from edsnlp.pipelines.misc.dates.patterns.relative import specific_dict


class Direction(Enum):

    FUTURE = "FUTURE"
    PAST = "PAST"
    CURRENT = "CURRENT"


class Mode(Enum):

    FROM = "FROM"
    UNTIL = "UNTIL"
    DURATION = "DURATION"


class BaseDate(BaseModel):

    mode: Optional[Mode] = None

    @root_validator(pre=True)
    def validate_strings(cls, d: Dict[str, str]) -> Dict[str, str]:
        result = d.copy()

        for k, v in d.items():
            if v is not None and "_" in k:
                key, value = k.split("_")
                result.update({key: value})

        return result


class AbsoluteDate(BaseDate):

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def parse(
        self,
        tz: Union[str, pendulum.tz.timezone] = "Europe/Paris",
        **kwargs,
    ) -> pendulum.datetime:

        if self.year and self.month and self.day:

            d = self.dict(exclude_none=True)

            d.pop("mode", None)

            return pendulum.datetime(**d, tz=tz)

        return None

    @validator("year")
    def validate_year(cls, v):
        if v > 100:
            return v

        if v < 25:
            return 2000 + v


class Relative(BaseDate):

    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    @root_validator(pre=True)
    def parse_unit(cls, d: Dict[str, str]) -> Dict[str, str]:
        """
        Units need to be handled separately.

        This validator modifies the key corresponding to the unit
        with the detected value

        Parameters
        ----------
        d : Dict[str, str]
            Original data

        Returns
        -------
        Dict[str, str]
            Transformed data
        """
        unit = d.get("unit")

        if unit:
            d[unit] = d.get("number")

        return d

    def parse(self, **kwargs) -> pendulum.duration:
        d = self.dict(exclude_none=True)

        direction = d.pop("direction", None)
        dir = -1 if direction == Direction.PAST else 1

        d.pop("mode", None)

        d = {f"{k}s": v for k, v in d.items()}

        td = dir * pendulum.duration(**d)
        return td


class RelativeDate(Relative):
    direction: Direction = Direction.CURRENT

    def parse(self, note_datetime: Optional[datetime] = None) -> pendulum.duration:
        td = super().parse()

        if note_datetime:
            return note_datetime + td

        return td

    @root_validator(pre=True)
    def handle_specifics(cls, d: Dict[str, str]) -> Dict[str, str]:
        """
        Specific patterns such as `aujourd'hui`, `hier`, etc,
        need to be handled separately.

        Parameters
        ----------
        d : Dict[str, str]
            Original data.

        Returns
        -------
        Dict[str, str]
            Modified data.
        """

        specific = d.get("specific")
        specific = specific_dict.get(specific)

        if specific:
            d.update(specific)

        return d


class Duration(Relative):
    mode: Mode = Mode.DURATION
