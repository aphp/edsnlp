from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, validator


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
