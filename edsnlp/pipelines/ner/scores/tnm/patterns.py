modifier_pattern = r"(?P<modifier>[cpyraum])"
tumour_pattern = r"t\s?(?P<tumour>([0-4o]|is|x))x?"
node_pattern = r"n\s?(?P<node>[0-3o]|x)x?"
metastasis_pattern = r"m\s?(?P<metastasis>[01o]|x)x?"

version_pattern = (
    r"\(?(?P<version>uicc|accj|tnm)"
    r"\s+([ée]ditions|[ée]d\.?)?\s*"
    r"(?P<version_year>\d{4}|\d{2})\)?"
)

spacer = r"(.|\n){1,5}"

tnm_pattern = f"(?<={version_pattern}{spacer})?"
tnm_pattern += modifier_pattern + r"\s*" + f"({tumour_pattern})"
tnm_pattern += r"\s*" + f"({node_pattern})?"
tnm_pattern += r"\s*" + f"({metastasis_pattern})?"
tnm_pattern += f"({spacer}{version_pattern})?"
