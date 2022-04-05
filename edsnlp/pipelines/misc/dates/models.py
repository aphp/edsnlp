from enum import Enum
from typing import Dict, Optional

import pendulum
from pydantic import BaseModel, root_validator, validator

from edsnlp.pipelines.misc.dates.patterns.relative import specific_dict


class Direction(Enum):

    future = 1
    past = -1
    since = -1
    after = 1
    during = 1
    until = 1


class BaseDate(BaseModel):

    direction: Optional[Direction] = None

    @root_validator(pre=True)
    def validate_strings(cls, d: Dict[str, str]) -> Dict[str, str]:
        result = d.copy()

        for k, v in d.items():
            if v is not None and "_" in k:
                key, value = k.split("_")
                result.update({key: value})
        return result

    @validator("direction", pre=True)
    def validate_direction(cls, v):
        if v is None:
            return v
        return Direction[v]


class AbsoluteDate(BaseDate):

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def parse(self) -> pendulum.datetime:

        if self.year and self.month and self.day:

            d = self.dict(exclude_none=True)
            d.pop("direction", None)

            return pendulum.datetime(**d, tz="Europe/Paris")

        return None


class RelativeDate(BaseDate):

    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def parse(self) -> pendulum.duration:
        d = self.dict(exclude_none=True)
        d.pop("direction", None)

        d = {f"{k}s": v for k, v in d.items()}

        return self.direction.value * pendulum.duration(**d)

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
