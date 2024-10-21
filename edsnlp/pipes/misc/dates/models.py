import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

import pydantic
from pandas._libs.tslibs.nattype import NaTType
from pydantic import BaseModel, Field, root_validator, validator
from pytz import timezone
from spacy.tokens import Span

from edsnlp.pipes.misc.dates.patterns.relative import specific_dict


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
    doc: Optional[Any] = Field(exclude=True, default=None)

    def norm(self) -> str:
        raise NotImplementedError()

    def to_datetime(self, **kwargs) -> Optional[datetime.datetime]:
        raise NotImplementedError()

    def to_duration(self, **kwargs) -> Optional[datetime.timedelta]:
        raise NotImplementedError()

    @validator("*", pre=True)
    def remove_space(cls, v):
        """Remove spaces. Useful for coping with ill-formatted PDF extractions."""
        if isinstance(v, str):
            return v.replace(" ", "")
        return v

    @property
    def datetime(self):
        return self.to_datetime()

    @property
    def duration(self):
        return self.to_duration()

    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict

    def __str__(self):
        return self.norm()


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
        note_datetime: Optional[Union[datetime.datetime]] = None,
        tz: Union[str, datetime.timezone] = None,
        infer_from_context: Optional[bool] = None,
        default_day=1,
        default_month=1,
        **kwargs,
    ) -> Optional[datetime.datetime]:
        """
        Convert the date to a datetime.datetime object.

        Parameters
        ----------
        tz : Optional[Union[str, pendulum.timezone]]
            The timezone to use. Defaults to None.
        note_datetime : Optional[Union[datetime.datetime, datetime.datetime]]
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
        Union[datetime.datetime, None]
        """

        if isinstance(tz, str):
            tz = timezone(tz)
        d = self.model_dump(exclude_none=True)
        d.pop("mode", None)
        d.pop("bound", None)

        if note_datetime is None and self.doc is not None:
            note_datetime = self.doc._.note_datetime

        if self.year and self.month and self.day:
            try:
                dt = (
                    tz.localize(datetime.datetime(**d))
                    if tz
                    else datetime.datetime(**d)
                )
                return dt
            except ValueError:
                return None
        elif (
            infer_from_context
            or infer_from_context is None
            and note_datetime is not None
        ):
            if note_datetime and not isinstance(note_datetime, NaTType):
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
                dt = (
                    tz.localize(datetime.datetime(**d))
                    if tz
                    else datetime.datetime(**d)
                )
                return dt
            except ValueError:
                return None

        return None

    def to_duration(
        self,
        note_datetime: Optional[Union[datetime.datetime, datetime.datetime]] = None,
        **kwargs,
    ) -> Optional[datetime.timedelta]:
        if note_datetime is None and self.doc is not None:
            note_datetime = self.doc._.note_datetime

        if note_datetime and not isinstance(note_datetime, NaTType):
            dt = self.to_datetime(note_datetime=note_datetime, **kwargs)
            delta = dt - note_datetime
            return delta
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

    def to_duration(self, note_datetime=None, **kwargs) -> datetime.timedelta:
        d = self.model_dump(exclude_none=True)

        direction = d.pop("direction", None)
        direction = -1 if direction == Direction.PAST else 1

        d.pop("mode", None)

        d = {f"{k}s": v for k, v in d.items()}

        if "months" in d:
            d["days"] = d.get("days", 0) + 30 * d.pop("months")

        if "years" in d:
            d["days"] = d.get("days", 0) + 365 * d.pop("years")

        td = direction * datetime.timedelta(**d)
        return td

    def to_datetime(self, **kwargs) -> Optional[datetime.datetime]:
        # for compatibility
        return None


class RelativeDate(Relative):
    direction: Direction = Direction.CURRENT

    def to_datetime(
        self,
        note_datetime: Optional[Union[datetime.datetime, datetime.datetime]] = None,
        **kwargs,
    ) -> Optional[datetime.datetime]:
        if note_datetime is None and self.doc is not None:
            note_datetime = self.doc._.note_datetime

        if note_datetime is not None and not isinstance(note_datetime, NaTType):
            d = self.model_dump(exclude_none=True)

            direction = d.pop("direction", None)
            dir = -1 if direction == Direction.PAST else 1

            d.pop("mode", None)
            d.pop("bound", None)

            d = {f"{k}s": v for k, v in d.items()}

            if "months" in d:
                d["days"] = d.get("days", 0) + 30 * d.pop("months")

            if "years" in d:
                d["days"] = d.get("days", 0) + 365 * d.pop("years")

            td = dir * datetime.timedelta(**d)

            return note_datetime + td

        return None

    def norm(self) -> str:
        if self.direction == Direction.CURRENT:
            d = self.model_dump(exclude_none=True)
            d.pop("direction", None)
            d.pop("mode", None)

            key = next(iter(d.keys()), "day")

            norm = f"~0 {key}"
        else:
            td = self.to_duration()
            norm = str(td)
            if td.total_seconds() > 0:
                norm = f"+{norm}"
            if norm.endswith(", 0:00:00"):
                norm = norm[:-9]

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
        td = self.to_duration()
        norm = f"during {td}"

        if norm.endswith(", 0:00:00"):
            norm = norm[:-9]

        return norm

    def to_duration(self, note_datetime=None, **kwargs) -> datetime.timedelta:
        d = self.model_dump(exclude_none=True)

        d = {f"{k}s": v for k, v in d.items() if k not in ("mode", "bound")}

        if "months" in d:
            d["days"] = d.get("days", 0) + 30 * d.pop("months")

        if "years" in d:
            d["days"] = d.get("days", 0) + 365 * d.pop("years")

        return datetime.timedelta(**d)
