from typing import Optional

from pydantic import BaseModel


class AdicapCode(BaseModel):
    code: str
    sampling: Optional[str]
    technic: Optional[str]
    organ: Optional[str]
    non_tumoral_pathology: Optional[str]
    tumoral_pathology: Optional[str]
    behaviour_type: Optional[str]

    def norm(self) -> str:
        return self.code

    def __str__(self):
        return self.norm()
