from typing import Optional

from pydantic import BaseModel


class AdicapCode(BaseModel):
    code: str
    sampling_mode: Optional[str] = None
    technic: Optional[str] = None
    organ: Optional[str] = None
    pathology: Optional[str] = None
    pathology_type: Optional[str] = None
    behaviour_type: Optional[str] = None

    def norm(self) -> str:
        return self.code

    def __str__(self):
        return self.norm()
