prefix_pattern = r"(?P<prefix>[cpPyraum]p?)"
tumour_pattern = r"T\s?(?P<tumour>([0-4o]|is))?(?P<tumour_specification>[abcdx]|mi)?"
tumour_pattern += r"(?:\((?P<tumour_suffix>[^()]{1,10})\))?"
node_pattern = r"(\s*\/?\s*([cpPyraum]p?)?\s*N\s?(?P<node>[0-3o]|x)"
node_pattern += (
    r"(?P<node_specification>[abcdx]|mi)?(?:\((?P<node_suffix>[^()]{1,10})\))?)"
)

metastasis_pattern = r"(\s*\/?\s*([cpPyraum]p?)?\s*M\s?(?P<metastasis>([01o]|x))x?)?"
resection_completeness = r"(\s*\/?\s*R\s?(?P<resection_completeness>[012]))?"

version_pattern = (
    r"\(?(?P<version>uicc|accj|tnm|UICC|ACCJ|TNM)"
    r"\s+([éeE]ditions|[éeE]d\.?)?\s*"
    r"(?P<version_year>\d{4}|\d{2})\)?"
)

spacer = r"(.|\n){1,5}"

tnm_pattern = f"(?<={version_pattern}{spacer})?"
tnm_pattern += prefix_pattern + r"\s*" + f"({tumour_pattern})"
tnm_pattern += r"\s*" + f"({node_pattern})?"
tnm_pattern += r"\s*" + f"({metastasis_pattern})?"
tnm_pattern += r"\s*" + f"({resection_completeness})?"
tnm_pattern += f"({spacer}{version_pattern})?"
tnm_pattern = r"(?:\b|^)" + tnm_pattern + r"(?:\b|$)"
