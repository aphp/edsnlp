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


class TNM(BaseModel):

    prefix: Optional[Prefix] = None
    tumour: Optional[Tumour] = None
    tumour_specification: Optional[Specification] = None
    tumour_suffix: Optional[str] = None
    node: Optional[Node] = None
    node_specification: Optional[Specification] = None
    node_suffix: Optional[str] = None
    metastasis: Optional[Metastasis] = None
    resection_completeness: Optional[int] = None
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

        if self.prefix is not None:
            norm.append(str(self.prefix))

        if (
            (self.tumour is not None)
            | (self.tumour_specification is not None)
            | (self.tumour_suffix is not None)
        ):
            norm.append(f"T{str(self.tumour or '')}")
            norm.append(f"{str(self.tumour_specification or '')}")
            norm.append(f"{str(self.tumour_suffix or '')}")

        if (
            (self.node is not None)
            | (self.node_specification is not None)
            | (self.node_suffix is not None)
        ):
            norm.append(f"N{str(self.node or '')}")
            norm.append(f"{str(self.node_specification or '')}")
            norm.append(f"{str(self.node_suffix or '')}")

        if self.metastasis is not None:
            norm.append(f"M{self.metastasis}")

        if self.resection_completeness is not None:
            norm.append(f"R{self.resection_completeness}")

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
        for k in set_keys.intersection(
            {
                "prefix",
                "tumour",
                "node",
                "metastasis",
                "tumour_specification",
                "node_specification",
                "tumour_suffix",
                "node_suffix",
            }
        ):
            v = d[k]
            if isinstance(v, TnmEnum):
                d[k] = v.value

        return d
