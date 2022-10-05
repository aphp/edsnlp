from typing import Optional

from pydantic import BaseModel


class AdicapCode(BaseModel):
    code: str
    sampling_mode: Optional[str]
    technic: Optional[str]
    organ: Optional[str]
    pathology: Optional[str]
    pathology_type: Optional[str]
    behaviour_type: Optional[str]

    def norm(self) -> str:
        return self.code

    def __str__(self):
        return self.norm()
