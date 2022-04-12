from typing import Optional, Union
from pydantic import BaseModel
from enum import Enum


class TnmEnum(Enum):
    def __str__(self) -> str:
        return self.value


class Modifier(TnmEnum):
    c = "c"
    p = "p"
    y = "y"
    a = "a"
    u = "u"
    m = "m"
    s = "s"


class Node(TnmEnum):
    x = "x"


class Tumour(TnmEnum):
    x = "x"
    i = "is"


class TNM(BaseModel):

    modifier: Optional[Union[int, Modifier]] = None
    tumour: Optional[Union[int, Tumour]] = None
    node: Optional[Union[int, Node]] = None
    metastasis: Optional[int] = None

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

        return "".join(norm)
