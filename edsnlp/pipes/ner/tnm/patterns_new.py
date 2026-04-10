'''
tumour_pattern = (
    r"(?P<tumour_prefix>[cpyramP]{1,2}\s?)?"  # Optional tumour prefix
    r"T\s?"  # 'T' followed by optional space
    r"(?P<tumour>([0-4]|is|[Xx]|[Oo]))"  # Tumour size (required if 'T' is present)
    r"(?:\s?(?P<tumour_specification>[abcdxm]|mi))?"  # Optional tumour specification
    r"(?:\s?\((?P<tumour_suffix>[^()]{1,10})\))?"  # Optional tumour suffix
)

node_pattern = (
    r"(?P<node_prefix>[cpyraP]{1,2}\s?)?"  # Optional node prefix
    r"N\s?"  # 'N' followed by optional space
    r"(?P<node>[Xx01234\+]|[Oo])"  # Node size/status (required if 'N' is present)
    r"(?:\s?(?P<node_specification>"
    r"[abcdxm]|mi|sn|i[-,+]|mol[-,+]|\(mi\)|\(sn\)|"
    r"\(i[-,+]\)|\(mol[-,+]\)|\(\d+\s*/\s*\d+\)))?"  # Optional specification
    r"(?:\s?\((?P<node_suffix>[^()]{1,10})\))?"  # Optional suffix
)

metastasis_pattern = (
    r"(?P<metastasis_prefix>[cpyraP]{1,2}\s?)?"  # Optional metastasis prefix
    r"M\s?"  # 'M' followed by optional space
    r"(?P<metastasis>[Xx0123\+]|[Oo])"  # Metastasis status (required if 'M' is present)
    r"(?:\s?(?P<metastasis_specification>"
    r"[abcdm]|i\+|mol\+|cy\+|\(i\+\)|\(mol\+\)|"
    r"\(cy\+\)|PUL|OSS|HEP|BRA|LYM|OTH|MAR|PLE|PER|ADR|SKI))?"  # Optional specification
)

pleura_pattern = (
    r"PL\s?(?P<pleura>([0123]|x))?"  # Optional pleura status (for lung cancer)
)

resection_pattern = (
    r"R\s?"
    r"(?P<resection>[Xx012\+])"  # Resection completeness
    r"(?:\s?(?P<resection_specification>is|cy\+|\(is\)|\(cy\+\)))?"  # Optional spec
    r"(?:\s?(?P<resection_loc>(\((?P<r_loc>[a-z]+)\)[,;\s]*)*))?"  # Optional loc
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
    r"(?!"
    r"(?:[cpyramP]{0,2}\s*)?"  # Optional prefix like p, yp, PT
    r"T\s*"
    r"(?:[0-4]|is|[xXoO])"  # T stage (includes is, x, o)
    r"(?:[abcdx]|mi)?"  # Optional specification
    r"(?:\s*\([^()]{1,10}\))?"  # Optional suffix
    r"(?:\s*[\s,\/\.\(\)]|$)"  # <-- KEY ADDITION: allow end-of-string ($)
    r"(?!\s*"
    + node_pattern
    + "?"
    + TNM_space
    + "?"
    + metastasis_pattern
    + "?"
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
    # + r"(?:\b|$|\n)"
)
'''




tumour_pattern = (
    r"(?P<tumour_prefix>[cpyraumsP]{1,2}\s*)?"  # Optional tumour prefix
    r"T\s*"  # 'T' followed by optional space
    r"(?P<tumour>([0-4]|is|[Xx]|[Oo]))"  # Tumour size (required if 'T' is present)
    r"(?:\s*(?P<tumour_specification>[abcdxm]|mi))?"  # Optional tumour specification
    r"(?:\s*\((?P<tumour_suffix>[^()]{1,20})\))?"  # Optional tumour suffix
)

node_pattern = (
    r"(?P<node_prefix>[cpyraumsP]{1,2}\s*)?"  # Optional node prefix
    r"N\s*"  # 'N' followed by optional space
    r"(?P<node>[Xx01234\+]|[Oo])"  # Node size/status (required if 'N' is present)
    r"(?:\s*(?P<node_specification>"
    r"[abcdxm]|mi|sn|i[-,+]|mol[-,+]|\(mi\)|\(sn\)|"
    r"\(i[-,+]\)|\(mol[-,+]\)|\(\d+\s*/\s*\d+\)))?"  # Optional specification
    r"(?:\s*\((?P<node_suffix>[^()]{1,20})\))?"  # Optional suffix
)

metastasis_pattern = (
    r"(?P<metastasis_prefix>[cpyraumsP]{1,2}\s*)?"  # Optional metastasis prefix
    r"M\s*"  # 'M' followed by optional space
    r"(?P<metastasis>[Xx0123\+]|[Oo])"  # Metastasis status (required if 'M' is present)
    r"(?:\s*(?P<metastasis_specification>"
    r"[abcdm]|i\+|mol\+|cy\+|\(i\+\)|\(mol\+\)|"
    r"\(cy\+\)|PUL|OSS|HEP|BRA|LYM|OTH|MAR|PLE|PER|ADR|SKI))?"  # Optional specification
    r"(?:\s*\((?P<metastasis_suffix>[^()]{1,20})\))?"  # Optional suffix
)

pleura_pattern = (
    r"PL\s*(?P<pleura>([0123]|x))?"  # Optional pleura status (for lung cancer)
)

resection_pattern = (
    r"(?P<resection_prefix>[cpyraumsP]{1,2}\s*)?"  # Optional metastasis prefix
    r"R\s*"
    r"(?P<resection>[Xx012\+])"  # Resection completeness
    r"(?:\s*(?P<resection_specification>is|cy\+|\(is\)|\(cy\+\)))?"  # Optional spec
    r"(?:\s*(?P<resection_loc>(\((?P<r_loc>[a-z]+)\)[,;\s]*)*))?"  # Optional loc
    r"(?:\s*\((?P<resection_suffix>[^()]{1,20})\))?"  # Optional suffix
)

TNM_space = r"(\s*[,\/]?\s*|\n)"  # Allow space, comma, or slash as delimiters
TNM_space = r"(?:\s*[,\/]?\s*|\n)"

logic_filter = (
    r"(?="
    # Conditions 1, 2, 3: If N, M, or R are present, the gatekeeper opens.
    # We use \b at the start to ensure we don't match the middle of a word.
    + r".*?\b" + node_pattern + r"|"
    + r".*?\b" + metastasis_pattern + r"|"
    + r".*?\b" + resection_pattern + r"|"
    
    # Condition 4: Standalone T (must have prefix AND spec)
    # We allow the spec to be followed by:
    # 1. A word boundary \b (space, punctuation, end of string)
    # 2. OR the start of the next component (N, M, R) to allow "glued" text.
    + r".*?\b[cpyraumsP]{1,2}\s*T\s*(?:[0-4]|is|[Xx]|[Oo])\s*(?:[abcd]|mi)(?=\b|[NMRP])"
    + r")"
)

logic_filter = (
    r"(?="
        # --- BRANCH 1: The "Qualified Solo" ---
        # Matches if T has a prefix AND a specification immediately.
        r"(?:[cpyraumsP]{1,2}\s*T\s*(?:[0-4]|is|[xo])\s*(?:[abcdxm]|mi)\b)"
        r"|"
        # --- BRANCH 2: The "T + NMR" ---
        # Matches any T (bare or prefixed) as long as it's followed by N, M, or R.
        r"(?:(?:[cpyraumsP]{0,2}\s*)?T\s*(?:[0-4]|is|[xo])" # T part
        r"(?:\s*(?:[abcdxm]|mi))?"                         # MISSING: Optional specification
        r"(?:\s*\([^()]{1,20}\))?"                        # MISSING: Optional suffix, e.g., (m)
        r"(?:\s*[,\/]?\s*|\n)"                            # TNM_space
        r"(?:[cpyraumsP]{0,2}\s*[NMR]\s*[x0-4\+o]))"            # Start of N, M, or R
    r")"
)

#banned_words = ["auto", "mtx"]
#ban_filter = r"(?!(?:" + "|".join(banned_words) + r")\b)"

tnm_pattern_new = (
    r"(?i)"             # Global Case-Insensitive
    r"(?:\b|^)"         # Boundary
    #+ ban_filter
    + logic_filter      # The "Gatekeeper"
    + r"(?P<T_component>" + tumour_pattern + r")"
    + TNM_space + "?"
    + r"(?P<N_component>" + node_pattern + r")?"
    + TNM_space + "?"
    + r"(?P<M_component>" + metastasis_pattern + r")?"
    + TNM_space + "?"
    + r"(?P<PL_component>" + pleura_pattern + r")?"
    + TNM_space + "?"
    + r"(?P<R_component>" + resection_pattern + r")?"
    + r"(?=[\s\(\)\.,;:/]|$)"
)
