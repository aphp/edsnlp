modifier_pattern = r"(?P<modifier>[cpPyraum]p?)"
tumour_pattern = r"T\s?(?P<tumour>([0-4o]|is))?(?P<tumour_specification>[abcdx])?"
node_pattern = r"\s*\/?\s*N\s?(?P<node>[0-3o]|x)(?P<node_specification>[abcdx])?"
metastasis_pattern = r"\s*\/?\s*M\s?(?P<metastasis>([01o]|x))x?"
resection_completeness = r"\s*\/?\s*R\s?(?P<resection_completeness>[012])"

version_pattern = (
    r"\(?(?P<version>uicc|accj|tnm|UICC|ACCJ|TNM)"
    r"\s+([éeE]ditions|[éeE]d\.?)?\s*"
    r"(?P<version_year>\d{4}|\d{2})\)?"
)

spacer = r"(.|\n){1,5}"

tnm_pattern = f"(?<={version_pattern}{spacer})?"
tnm_pattern += modifier_pattern + r"\s*" + f"({tumour_pattern})"
tnm_pattern += r"\s*" + f"({node_pattern})?"
tnm_pattern += r"\s*" + f"({metastasis_pattern})?"
tnm_pattern += r"\s*" + f"({resection_completeness})?"
tnm_pattern += f"({spacer}{version_pattern})?"
tnm_pattern = r"\b" + tnm_pattern + r"\b"
