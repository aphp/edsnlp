from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union

import pendulum
from pydantic import BaseModel, root_validator, validator
from spacy.tokens import Span

from edsnlp.pipelines.misc.dates.patterns.relative import specific_dict


class Direction(Enum):

    FUTURE = "FUTURE"
    PAST = "PAST"
    CURRENT = "CURRENT"


class Mode(Enum):

    FROM = "FROM"
    UNTIL = "UNTIL"
    DURATION = "DURATION"


class Period(BaseModel):
    FROM: Optional[Span] = None
    UNTIL: Optional[Span] = None
    DURATION: Optional[Span] = None

    class Config:
        arbitrary_types_allowed = True


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

    def to_datetime(
        self,
        tz: Union[str, pendulum.tz.timezone] = "Europe/Paris",
        **kwargs,
    ) -> Optional[pendulum.datetime]:

        if self.year and self.month and self.day:

            d = self.dict(exclude_none=True)

            d.pop("mode", None)

            return pendulum.datetime(**d, tz=tz)

        return None

    def norm(self) -> str:

        year = str(self.year) if self.year else "????"
        month = f"{self.month:02}" if self.month else "??"
        day = f"{self.day:02}" if self.day else "??"

        norm = "-".join([year, month, day])

        if self.hour:
            norm += f" {self.hour:02}h"

        if self.minute:
            norm += f"{self.minute:02}m"

        if self.second:
            norm += f"{self.second:02}s"

        return norm

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

    def to_datetime(self, **kwargs) -> pendulum.Duration:
        d = self.dict(exclude_none=True)

        direction = d.pop("direction", None)
        dir = -1 if direction == Direction.PAST else 1

        d.pop("mode", None)

        d = {f"{k}s": v for k, v in d.items()}

        td = dir * pendulum.duration(**d)
        return td


class RelativeDate(Relative):
    direction: Direction = Direction.CURRENT

    def to_datetime(
        self, note_datetime: Optional[datetime] = None
    ) -> pendulum.Duration:
        td = super(RelativeDate, self).to_datetime()

        if note_datetime is not None:
            return note_datetime + td

        return td

    def norm(self) -> str:

        if self.direction == Direction.CURRENT:
            d = self.dict(exclude_none=True)
            d.pop("direction")

            (key,) = d.keys()

            norm = f"~0 {key}"
        else:
            td = self.to_datetime()
            norm = str(td)
            if td.in_seconds() > 0:
                norm = f"+{norm}"

        return norm

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

    def norm(self) -> str:

        td = self.to_datetime()
        return f"during {td}"
