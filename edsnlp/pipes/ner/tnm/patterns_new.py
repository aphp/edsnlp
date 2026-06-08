tumour_pattern = (
    r"(?P<tumour_prefix>[cpyraumsP]{1,2}\s*)?"
    r"T\s*"
    r"(?P<tumour>([0-4]|is|[Xx]|[Oo]))"
    # mi before m, and m uses negative lookahead to avoid consuming M from M_component
    r"(?:\s*(?P<tumour_specification>[abcdx]|mi|m(?!\s*[0-9xXoO+])))?"
    r"(?:\s*\((?P<tumour_suffix>[^()]{1,20})\))?"
)

node_pattern = (
    r"(?P<node_prefix>[cpyraumsP]{1,2}\s*)?"
    r"N\s*"
    r"(?P<node>[Xx01234\+]|[Oo])"
    # mi before m, and m uses negative lookahead to avoid consuming M from M_component
    r"(?:\s*(?P<node_specification>"
    r"[abcdx]|mi|m(?!\s*[0-9xXoO+])|sn|i[-,+]|mol[-,+]|\(mi\)|\(sn\)|"
    r"\(i[-,+]\)|\(mol[-,+]\)|\(\d+\s*/\s*\d+\)))?"
    r"(?:\s*\((?P<node_suffix>[^()]{1,20})\))?"
)

metastasis_pattern = (
    r"(?P<metastasis_prefix>[cpyraumsP]{1,2}\s*)?"
    r"M\s*"
    r"(?P<metastasis>[Xx0123\+]|[Oo])"
    r"(?:\s*(?P<metastasis_specification>"
    r"[abcdm]|i\+|mol\+|cy\+|\(i\+\)|\(mol\+\)|"
    r"\(cy\+\)|PUL|OSS|HEP|BRA|LYM|OTH|MAR|PLE|PER|ADR|SKI))?"
    r"(?:\s*\((?P<metastasis_suffix>[^()]{1,20})\))?"
)

pleura_pattern = (
    r"PL\s*(?P<pleura>([0123]|x))?"
)

resection_pattern = (
    r"(?P<resection_prefix>[cpyraumsP]{1,2}\s*)?"
    r"R\s*"
    r"(?P<resection>[Xx012\+])"
    r"(?:\s*(?P<resection_specification>is|cy\+|\(is\)|\(cy\+\)))?"
    r"(?:\s*(?P<resection_loc>(\((?P<r_loc>[a-z]+)\)[,;\s]*)*))?"
    r"(?:\s*\((?P<resection_suffix>[^()]{1,20})\))?"
)

TNM_space = r"(?:\s*[,\/]?\s*|\n)"

logic_filter = (
    r"(?="
        # --- BRANCH 1: Standalone T with prefix AND specification ---
        r"(?:[cpyraumsP]{1,2}\s*T\s*(?:[0-4]|is|[xo])\s*(?:[abcdxm]|mi)\b)"
        r"|"
        # --- BRANCH 2: T followed by N, M, or R ---
        r"(?:(?:[cpyraumsP]{0,2}\s*)?T\s*(?:[0-4]|is|[xo])"
        r"(?:\s*(?:[abcdxm]|mi))?"
        r"(?:\s*\([^()]{1,20}\))?"
        r"(?:\s*[,\/]?\s*|\n)"
        r"(?:[cpyraumsP]{0,2}\s*[NMR]\s*[x0-4\+o]))"
    r")"
)

tnm_pattern_new = (
    r"(?i)"
    r"(?:\b|^)"
    + logic_filter
    + r"(?P<T_component>" + tumour_pattern + r")"
    # Each optional component is grouped with its preceding TNM_space so that
    # trailing whitespace is not greedily consumed when the component is absent.
    + r"(?:" + TNM_space + r"(?P<N_component>" + node_pattern + r"))?"
    + r"(?:" + TNM_space + r"(?P<M_component>" + metastasis_pattern + r"))?"
    + r"(?:" + TNM_space + r"(?P<PL_component>" + pleura_pattern + r"))?"
    + r"(?:" + TNM_space + r"(?P<R_component>" + resection_pattern + r"))?"
    + r"(?=[\s\(\)\.,;:/]|$)"
)
