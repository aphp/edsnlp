import warnings
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import pydantic
from pydantic import field_validator

if TYPE_CHECKING:
    from pydantic.typing import (
        AbstractSetIntStr,
        DictStrAny,
        MappingIntStrAny,
    )


def validator(x: str, allow_reuse=True, pre=False):
    return field_validator(x, mode="before" if pre else "after")


class TnmEnum(Enum):
    def __str__(self) -> str:
        return self.value


class Prefix(TnmEnum):
    clinical = "c"
    histopathology = "p"
    histopathology2 = "P"
    neoadjuvant_therapy = "y"
    recurrent = "r"
    autopsy = "a"
    ultrasonography = "u"
    multifocal = "m"
    py = "yp"
    mp = "mp"


class Tumour(TnmEnum):
    unknown = "x"
    in_situ = "is"
    score_0 = "0"
    score_1 = "1"
    score_2 = "2"
    score_3 = "3"
    score_4 = "4"
    o = "o"


class Specification(TnmEnum):
    a = "a"
    b = "b"
    c = "c"
    d = "d"
    mi = "mi"
    x = "x"


class Node(TnmEnum):
    unknown = "x"
    score_0 = "0"
    score_1 = "1"
    score_2 = "2"
    score_3 = "3"
    o = "o"


class Metastasis(TnmEnum):
    unknown = "x"
    score_0 = "0"
    score_1 = "1"
    o = "o"
    score_1x = "1x"
    score_2x = "2x"
    ox = "ox"


class TNM(pydantic.BaseModel):
    tumour_prefix: Optional[str] = None
    tumour: Optional[str] = None
    tumour_specification: Optional[str] = None
    tumour_suffix: Optional[str] = None
    node_prefix: Optional[str] = None
    node: Optional[str] = None
    node_specification: Optional[str] = None
    node_suffix: Optional[str] = None
    metastasis_prefix: Optional[str] = None
    metastasis: Optional[str] = None
    metastasis_specification: Optional[str] = None
    pleura: Optional[str] = None
    resection: Optional[str] = None
    resection_specification: Optional[str] = None
    resection_loc: Optional[str] = None
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

        if self.tumour_prefix:
            norm.append(f"{self.tumour_prefix or ''}")

        if self.tumour:
            norm.append(f"T{self.tumour}")
            if self.tumour_specification:
                norm.append(f"{self.tumour_specification or ''}")
            if self.tumour_suffix:
                norm.append(f"{self.tumour_suffix or ''}")

        if self.node_prefix:
            norm.append(f"{self.node_prefix or ''}")

        if self.node:
            norm.append(f"N{self.node}")
            if self.node_specification:
                norm.append(f"{self.node_specification or ''}")
            if self.node_suffix:
                norm.append(f"{self.node_suffix or ''}")

        if self.metastasis_prefix:
            norm.append(f"{self.metastasis_prefix or ''}")

        if self.metastasis:
            norm.append(f"M{self.metastasis}")
            if self.metastasis_specification:
                norm.append(f"{self.metastasis_specification or ''}")

        if self.pleura:
            norm.append(f"PL{self.pleura}")

        if self.resection:
            norm.append(f"R{self.resection}")
            if self.resection_specification:
                norm.append(f"{self.resection_specification or ''}")
            if self.resection_loc:
                norm.append(f"{self.resection_loc or ''}")

        if self.version is not None and self.version_year is not None:
            norm.append(f" ({self.version.upper()} {self.version_year})")

        return "".join(norm)

    def __str__(self):
        return self.norm()

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

        d = self.model_dump(
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        set_keys = set(d.keys())
        for k in set_keys.intersection(
            {
                "tumour_prefix",
                "tumour",
                "tumour_specification",
                "tumour_suffix",
                "node_prefix",
                "node",
                "node_specification",
                "node_suffix",
                "metastasis_prefix",
                "metastasis",
                "metastasis_specification",
                "pleura",
                "resection",
                "resection_specification",
                "resection_loc",
            }
        ):
            v = d[k]
            if isinstance(v, TnmEnum):
                d[k] = v.value

        return d
