import datetime
from enum import Enum
from typing import Dict, Optional, Union

import pendulum
from pandas._libs.tslibs.nattype import NaTType
from pydantic import BaseModel, root_validator, validator
from spacy.tokens import Span

from edsnlp.pipelines.misc.dates.patterns.relative import specific_dict


class Direction(str, Enum):
    FUTURE = "future"
    PAST = "past"
    CURRENT = "current"


class Bound(str, Enum):
    UNTIL = "until"
    FROM = "from"


class Mode(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    DURATION = "duration"


class Period(BaseModel):
    FROM: Optional[Span] = None
    UNTIL: Optional[Span] = None
    DURATION: Optional[Span] = None

    class Config:
        arbitrary_types_allowed = True


class BaseDate(BaseModel):
    mode: Mode = None
    bound: Optional[Bound] = None

    @validator("*", pre=True)
    def remove_space(cls, v):
        """Remove spaces. Useful for coping with ill-formatted PDF extractions."""
        if isinstance(v, str):
            return v.replace(" ", "")
        return v

    @root_validator(pre=True)
    def validate_strings(cls, d: Dict[str, str]) -> Dict[str, str]:
        result = d.copy()

        for k, v in d.items():
            if v is not None and "_" in k:
                key, value = k.split("_")
                result.update({key: value})

        return result


class AbsoluteDate(BaseDate):
    mode: Mode = Mode.ABSOLUTE
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def to_datetime(
        self,
        note_datetime: Optional[Union[pendulum.datetime, datetime.datetime]] = None,
        tz: Union[str, pendulum.tz.timezone] = "Europe/Paris",
        infer_from_context: Optional[bool] = None,
        default_day=1,
        default_month=1,
        **kwargs,
    ) -> Optional[pendulum.datetime]:
        """
        Convert the date to a pendulum.datetime object.

        Parameters
        ----------
        tz : Optional[Union[str, pendulum.tz.timezone]]
            The timezone to use. Defaults to "Europe/Paris".
        note_datetime : Optional[Union[pendulum.datetime, datetime.datetime]]
            The datetime of the note. Used to infer missing parts of the date.
        infer_from_context : bool
            Whether to infer missing parts of the date from the note datetime.
            In a (year, month, day) triplet:

                - if only year is missing, it will be inferred from the note datetime
                - if only month is missing, it will be inferred from the note datetime
                - if only day is missing, it will be set to `default_day`
                - if only the year is given, the day and month will be set to
                  `default_day` and `default_month`
                - if only the month is given, the day will be set to `default_day`
                  and the year will be inferred from the note datetime
                - if only the day is given, the month and year will be inferred from
                  the note datetime
        default_day : int
            Default day to use when inferring missing parts of the date.
        default_month : int
            Default month to use when inferring missing parts of the date.

        Returns
        -------
        Optional[pendulum.datetime]
        """

        d = self.dict(exclude_none=True)
        d.pop("mode", None)
        d.pop("bound", None)

        if self.year and self.month and self.day:
            try:
                return pendulum.datetime(**d, tz=tz)
            except ValueError:
                return None
        elif (
            infer_from_context
            or infer_from_context is None
            and note_datetime is not None
        ):
            if note_datetime and not isinstance(note_datetime, NaTType):
                note_datetime = pendulum.instance(note_datetime)

                if self.year is None:
                    d["year"] = note_datetime.year
                if self.month is None:
                    if self.day is None:
                        d["month"] = default_month
                    else:
                        d["month"] = note_datetime.month
                if self.day is None:
                    d["day"] = default_day
            else:
                if self.year is None:
                    return None
                if self.month is None:
                    d["month"] = default_month
                if self.day is None:
                    d["day"] = default_day

            try:
                return pendulum.datetime(**d, tz=tz)
            except ValueError:
                return None

        return None

    def to_duration(
        self,
        note_datetime: Optional[Union[pendulum.datetime, datetime.datetime]] = None,
        **kwargs,
    ) -> Optional[pendulum.Duration]:

        if note_datetime and not isinstance(note_datetime, NaTType):
            note_datetime = pendulum.instance(note_datetime)
            dt = self.to_datetime(note_datetime=note_datetime, **kwargs)
            delta = dt.diff(note_datetime)
            return delta.as_interval()
        else:
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

    def __str__(self):
        return self.norm()


class Relative(BaseDate):
    mode: Mode = Mode.RELATIVE
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

    def to_duration(self, note_datetime=None, **kwargs) -> pendulum.Duration:
        d = self.dict(exclude_none=True)

        direction = d.pop("direction", None)
        dir = -1 if direction == Direction.PAST else 1

        d.pop("mode", None)

        d = {f"{k}s": v for k, v in d.items()}

        td = dir * pendulum.duration(**d)
        return td

    def to_datetime(self, **kwargs) -> Optional[pendulum.datetime]:
        # for compatibility
        return None


class RelativeDate(Relative):
    direction: Direction = Direction.CURRENT

    def to_datetime(
        self,
        note_datetime: Optional[Union[pendulum.datetime, datetime.datetime]] = None,
        **kwargs,
    ) -> Optional[pendulum.datetime]:

        if note_datetime is not None and not isinstance(note_datetime, NaTType):
            note_datetime = pendulum.instance(note_datetime)

            d = self.dict(exclude_none=True)

            direction = d.pop("direction", None)
            dir = -1 if direction == Direction.PAST else 1

            d.pop("mode", None)
            d.pop("bound", None)

            d = {f"{k}s": v for k, v in d.items()}

            td = dir * pendulum.duration(**d)

            return note_datetime + td

        return None

    def norm(self) -> str:

        if self.direction == Direction.CURRENT:
            d = self.dict(exclude_none=True)
            d.pop("direction", None)
            d.pop("mode", None)

            key = next(iter(d.keys()), "day")

            norm = f"~0 {key}"
        else:
            td = self.to_duration()
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

    def __str__(self):
        return self.norm()


class Duration(Relative):
    mode: Mode = Mode.DURATION

    def norm(self) -> str:
        td = self.to_duration()
        return f"during {td}"

    def to_duration(self, note_datetime=None, **kwargs) -> pendulum.Duration:
        d = self.dict(exclude_none=True)

        d = {f"{k}s": v for k, v in d.items() if k not in ("mode", "bound")}

        return pendulum.duration(**d)
