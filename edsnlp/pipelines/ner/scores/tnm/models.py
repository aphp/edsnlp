from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, validator

if TYPE_CHECKING:
    from pydantic.typing import (
        AbstractSetIntStr,
        DictStrAny,
        MappingIntStrAny,
    )

import warnings


class TnmEnum(Enum):
    def __str__(self) -> str:
        return self.value


class Unknown(TnmEnum):
    unknown = "x"


class Modifier(TnmEnum):
    clinical = "c"
    histopathology = "p"
    neoadjuvant_therapy = "y"
    recurrent = "r"
    autopsy = "a"
    ultrasonography = "u"
    multifocal = "m"


class Tumour(TnmEnum):
    unknown = "x"
    in_situ = "is"


class TNM(BaseModel):

    modifier: Optional[Union[int, Modifier]] = None
    tumour: Optional[Union[int, Tumour]] = None
    node: Optional[Union[int, Unknown]] = None
    metastasis: Optional[Union[int, Unknown]] = None

    version: Optional[str] = None
    version_year: Optional[int] = None

    @validator("*", pre=True)
    def coerce_o(cls, v):
        if isinstance(v, str):
            v = v.replace("o", "0")
        return v

    @validator("version_year")
    def validate_year(cls, v):
        if v is None:
            return v

        if v < 40:
            v += 2000
        elif v < 100:
            v += 1900

        return v

    def norm(self) -> str:
        norm = []

        if self.modifier is not None:
            norm.append(str(self.modifier))

        if self.tumour is not None:
            norm.append(f"T{self.tumour}")

        if self.node is not None:
            norm.append(f"N{self.node}")

        if self.metastasis is not None:
            norm.append(f"M{self.metastasis}")

        if self.version is not None and self.version_year is not None:
            norm.append(f" ({self.version.upper()} {self.version_year})")

        return "".join(norm)

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> "DictStrAny":
        """
        Generate a dictionary representation of the model,
        optionally specifying which fields to include or exclude.

        """
        if skip_defaults is not None:
            warnings.warn(
                f"""{self.__class__.__name__}.dict(): "skip_defaults"
                is deprecated and replaced by "exclude_unset" """,
                DeprecationWarning,
            )
            exclude_unset = skip_defaults

        d = dict(
            self._iter(
                to_dict=True,
                by_alias=by_alias,
                include=include,
                exclude=exclude,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )
        )
        set_keys = set(d.keys())
        for k in set_keys.intersection({"modifier", "tumour", "node", "metastasis"}):
            v = d[k]
            if isinstance(v, int):
                d[k] = {
                    f"{k}_string": None,
                    f"{k}_int": v,
                }
            elif isinstance(v, TnmEnum):
                d[k] = {
                    f"{k}_string": v.value,
                    f"{k}_int": None,
                }
            else:
                d[k] = {
                    f"{k}_string": None,
                    f"{k}_int": None,
                }

        return d
