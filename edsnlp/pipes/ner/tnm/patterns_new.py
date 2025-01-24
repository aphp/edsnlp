tumour_pattern = (
    r"(?P<tumour_prefix>[cpyramP]{1,2}\s?)?"  # Optional tumour prefix
    r"T\s?"  # 'T' followed by optional space
    r"(?P<tumour>([0-4]|is|[Xx]|[Oo]))"  # Tumour size (required if 'T' is present)
    r"(?:\s?(?P<tumour_specification>[abcdx]|mi))?"  # Optional tumour specification
    r"(?:\s?\((?P<tumour_suffix>[^()]{1,10})\))?"  # Optional tumour suffix
)

node_pattern = (
    r"(?P<node_prefix>[cpyraP]{1,2}\s?)?"  # Optional node prefix
    r"N\s?"  # 'N' followed by optional space
    r"(?P<node>[Xx01234\+]|[Oo])"  # Node size/status (required if 'N' is present)
    r"(?:\s?(?P<node_specification>"
    r"[abcdx]|mi|sn|i[-,+]|mol[-,+]|\(mi\)|\(sn\)|"
    r"\(i[-,+]\)|\(mol[-,+]\)|\(\d+\s*/\s*\d+\)))?"  # Optional specification
    r"(?:\s?\((?P<node_suffix>[^()]{1,10})\))?"  # Optional suffix
)

metastasis_pattern = (
    r"(?P<metastasis_prefix>[cpyraP]{1,2}\s?)?"  # Optional metastasis prefix
    r"M\s?"  # 'M' followed by optional space
    r"(?P<metastasis>[Xx0123\+]|[Oo])"  # Metastasis status (required if 'M' is present)
    r"(?:\s?(?P<metastasis_specification>"
    r"[abcd]|i\+|mol\+|cy\+|\(i\+\)|\(mol\+\)|"
    r"\(cy\+\)|PUL|OSS|HEP|BRA|LYM|OTH|MAR|PLE|PER|ADR|SKI))?"  # Optional specification
)

pleura_pattern = (
    r"PL\s?(?P<pleura>([0123]|x))?"  # Optional pleura status (for lung cancer)
)

resection_pattern = (
    r"R\s?"
    r"(?P<resection>[Xx012]|[Oo])?" # Optional resection completeness
    r"(?:\s?(?P<resection_specification>is|cy\+|\(is\)|\(cy\+\)))?"  # Optional specification
    r"(?:\s?(?P<resection_loc>(\((?P<r_loc>[a-z]+)\)[,;\s]*)*))?"  # Optional localization with space
)

version_pattern = (
    r"\(?(?P<version>uicc|accj|tnm|UICC|ACCJ|TNM)"  # TNM version
    r"\s+([éeE]ditions|[éeE]d\.?)?\s*"
    r"(?P<version_year>\d{4}|\d{2})\)?"  # Year of the version
)

TNM_space = r"(\s*[,\/]?\s*|\n)"  # Allow space, comma, or slash as delimiters

# We need te exclude pattern like 'T1', 'T2' if they are not followed by node or
# metastasis sections.

exclude_pattern = (
    r"(?!T\s*[0-4]\s*[.,\/](?!\s*"
    + node_pattern
    + "?"
    + TNM_space
    + "?"
    + metastasis_pattern
    + "?"
    + "))"
)

exclude_pattern = (
    r"(?!"
        r"(?:[cpyramP]{0,2}\s*)?"        # Optional prefix like p, yp, PT
        r"T\s*"
        r"(?:[0-4]|is|[xXoO])"           # T stage (includes is, x, o)
        r"(?:[abcdx]|mi)?"               # Optional specification
        r"(?:\s*\([^()]{1,10}\))?"       # Optional suffix
        r"(?:\s*[\s,\/\.\(\)]|$)"        # <-- KEY ADDITION: allow end-of-string ($)
        r"(?!\s*"
            + node_pattern + "?" + TNM_space + "?" + metastasis_pattern + "?"
        + ")"
    + ")"
)

tnm_pattern_new = (
    r"(?:\b|^)"
    + exclude_pattern
    + r"(?:"
    + r"(?P<T_component>"
    + tumour_pattern
    + ")"
    + TNM_space
    + "?"
    + r"(?P<N_component>"
    + node_pattern
    + ")?"
    + TNM_space
    + "?"
    + r"(?P<M_component>"
    + metastasis_pattern
    + ")?"
    + TNM_space
    + "?"
    + r"(?P<PL_component>"
    + pleura_pattern
    + ")?"
    + TNM_space
    + "?"
    + r"(?P<R_component>"
    + resection_pattern
    + ")?"
    + TNM_space
    + "?"
    + r"(?P<V_component>"
    + version_pattern
    + ")?"
    + r")"
    + r"(?=[\s\(\)\.,;:/]|$)"
    #+ r"(?:\b|$|\n)"
)
