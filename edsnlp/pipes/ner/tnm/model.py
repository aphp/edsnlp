import re
import warnings
from typing import TYPE_CHECKING, Optional, Union

import pydantic
from pydantic import field_validator
from typing_extensions import deprecated

if TYPE_CHECKING:
    from pydantic.typing import (
        AbstractSetIntStr,
        DictStrAny,
        MappingIntStrAny,
    )

# Fields holding a numeric stage, where the letter `o` is a typo for the digit
# `0`. The coercion must NOT be applied to the other fields: they hold free
# text (`mol+`, `OSS`, `(foie)`, ...) where an `o` is a genuine letter.
SCORE_FIELDS = frozenset(
    {"tumour", "node", "metastasis", "pleura", "resection"},
)

# A parenthesised suffix is only part of the canonical form when it reads as a
# TNM qualifier (`(m)` multifocal, `(sn)` sentinel node, ...). The capture
# groups are deliberately permissive, so anything else -- a classification
# year, a free-text comment -- is kept in the field but left out of `norm()`.
TNM_QUALIFIER = re.compile(r"[A-Za-z]{1,3}$")


def validator(*fields: str, allow_reuse=True, pre=False):
    return field_validator(*fields, mode="before" if pre else "after")


class TNM(pydantic.BaseModel):
    """Structured representation of a parsed TNM staging mention.

    All fields store the raw text captured by the regex (case-preserved),
    except `version_year` which is normalised to a four-digit integer.
    Parenthesised specifications such as `(sn)` are kept with their
    parentheses in the field value; `norm()` strips them for the canonical
    form. The letter `o` is normalised to `0` in the numeric stage fields
    only (see `SCORE_FIELDS`).
    """

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
    metastasis_suffix: Optional[str] = None
    pleura: Optional[str] = None
    resection_prefix: Optional[str] = None
    resection: Optional[str] = None
    resection_specification: Optional[str] = None
    resection_loc: Optional[str] = None
    resection_suffix: Optional[str] = None
    version: Optional[str] = None
    version_year: Optional[int] = None

    @validator(*sorted(SCORE_FIELDS), pre=True)
    def coerce_o(cls, v):
        if isinstance(v, str):
            v = v.replace("o", "0").replace("O", "0")
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

    @property
    @deprecated("`prefix` is deprecated, use `tumour_prefix` instead")
    def prefix(self) -> Optional[str]:
        """Deprecated alias for `tumour_prefix`."""
        return self.tumour_prefix

    @property
    @deprecated("`resection_completeness` is deprecated, use `resection` instead")
    def resection_completeness(self) -> Optional[Union[int, str]]:
        """Deprecated alias for `resection`.

        The field used to be an `int`, so a numeric status is returned as one.
        The values the previous model could not represent (`x`, `+`) are
        returned as strings.
        """
        v = self.resection
        return int(v) if v is not None and v.isdigit() else v

    @staticmethod
    def _norm_str(v: Optional[str]) -> str:
        """Strip surrounding whitespace and parentheses from captured values."""
        if not v:
            return ""
        v = v.strip()
        if v.startswith("(") and v.endswith(")"):
            v = v[1:-1]
        return v

    @classmethod
    def _norm_suffix(cls, v: Optional[str]) -> str:
        """Keep a parenthesised suffix in `norm()` only if it is a qualifier."""
        v = cls._norm_str(v)
        return v if TNM_QUALIFIER.match(v) else ""

    def norm(self) -> str:
        norm = []

        if self.tumour_prefix:
            norm.append(self._norm_str(self.tumour_prefix))

        if self.tumour:
            norm.append(f"T{self.tumour}")
            if self.tumour_specification:
                norm.append(self._norm_str(self.tumour_specification))
            if self.tumour_suffix:
                norm.append(self._norm_suffix(self.tumour_suffix))

        if self.node_prefix:
            norm.append(self._norm_str(self.node_prefix))

        if self.node:
            norm.append(f"N{self.node}")
            if self.node_specification:
                norm.append(self._norm_str(self.node_specification))
            if self.node_suffix:
                norm.append(self._norm_suffix(self.node_suffix))

        if self.metastasis_prefix:
            norm.append(self._norm_str(self.metastasis_prefix))

        if self.metastasis:
            norm.append(f"M{self.metastasis}")
            if self.metastasis_specification:
                norm.append(self._norm_str(self.metastasis_specification))
            if self.metastasis_suffix:
                norm.append(self._norm_suffix(self.metastasis_suffix))

        if self.pleura:
            norm.append(f"PL{self.pleura}")

        if self.resection_prefix:
            norm.append(self._norm_str(self.resection_prefix))

        if self.resection:
            norm.append(f"R{self.resection}")
            if self.resection_specification:
                norm.append(self._norm_str(self.resection_specification))
            if self.resection_loc:
                norm.append(self._norm_str(self.resection_loc))
            if self.resection_suffix:
                norm.append(self._norm_suffix(self.resection_suffix))

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

        return self.model_dump(
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
