"""Regular expression used by the `eds.tnm` pipe.

The pattern is assembled from one sub-pattern per TNM component (T, N, M, PL,
R) plus an optional trailing classification version, so that each piece can be
read and amended in isolation. The final expression is case-insensitive.
"""

# Modifier prefixes shared by every component: c(linical), p(athological),
# y(post-therapy), r(ecurrent), a(utopsy), u(ltrasonography), m(ultifocal) and
# s(urgical). Up to two may be combined (`yp`, `mp`, ...). The whole pattern is
# case-insensitive, so uppercase variants such as `P` are covered as well.
prefix_pattern = r"[cpyraums]{1,2}\s*"

# Guard preventing a trailing classification tag such as `(UICC 2017)` from
# being swallowed by a component suffix: it is matched by `version_pattern`.
not_version = r"(?!\s*(?:uicc|ajcc|accj|tnm)\b)"

# Separator between a stage value and its specification. The specification may
# be detached from its value (`pT2 b`), but a letter glued to a following
# N/M/R stage is that component's prefix, not this one's specification: in
# `pT1 cN1 M0` the `c` qualifies N, whereas in `pT2c N0` it qualifies T.
spec_space = r"(?:\s+(?!(?:" + prefix_pattern + r")?[NMR]\s*[x0-4+o]))?"

tumour_pattern = (
    r"(?P<tumour_prefix>" + prefix_pattern + r")?"
    r"T\s*"
    r"(?P<tumour>[0-4]|is|x|o)"
    # `mi` is listed before `m`, and `m` uses a negative lookahead so that the
    # `M` of the metastasis component is never consumed as a T specification.
    r"(?:" + spec_space + r"(?P<tumour_specification>[abcdx]|mi|m(?!\s*[0-9xo+])))?"
    r"(?:\s*\(" + not_version + r"(?P<tumour_suffix>[^()]{1,20})\))?"
)

node_pattern = (
    r"(?P<node_prefix>" + prefix_pattern + r")?"
    r"N\s*"
    r"(?P<node>[x0-4+o])"
    # Same `mi` / `m` ordering and lookahead as for the tumour component.
    r"(?:" + spec_space + r"(?P<node_specification>"
    r"[abcdx]|mi|m(?!\s*[0-9xo+])|sn|i[-+]|mol[-+]|\(mi\)|\(sn\)|"
    r"\(i[-+]\)|\(mol[-+]\)|\(\d+\s*/\s*\d+\)))?"
    r"(?:\s*\(" + not_version + r"(?P<node_suffix>[^()]{1,20})\))?"
)

metastasis_pattern = (
    r"(?P<metastasis_prefix>" + prefix_pattern + r")?"
    r"M\s*"
    r"(?P<metastasis>[x0-3+o])"
    r"(?:" + spec_space + r"(?P<metastasis_specification>"
    r"[abcdm]|i\+|mol\+|cy\+|\(i\+\)|\(mol\+\)|"
    r"\(cy\+\)|PUL|OSS|HEP|BRA|LYM|OTH|MAR|PLE|PER|ADR|SKI))?"
    r"(?:\s*\(" + not_version + r"(?P<metastasis_suffix>[^()]{1,20})\))?"
)

pleura_pattern = r"PL\s*(?P<pleura>[0-3]|x)"

resection_pattern = (
    r"(?P<resection_prefix>" + prefix_pattern + r")?"
    r"R\s*"
    r"(?P<resection>[x012+])"
    r"(?:\s*(?P<resection_specification>is|cy\+|\(is\)|\(cy\+\)))?"
    # One or more parenthesised single-word locations, e.g. `R1(foie)(poumon)`.
    r"(?:\s*(?P<resection_loc>(?:\([a-z]+\)[,;\s]*)+))?"
    r"(?:\s*\(" + not_version + r"(?P<resection_suffix>[^()]{1,20})\))?"
)

version_pattern = (
    r"\(?\s*(?P<version>uicc|ajcc|accj|tnm)"
    r"\s*(?:\d{1,2}\s*(?:[èe]me|[èe]re|e|è|er|th|nd|rd|st)\s*)?"
    r"(?:[ée]ditions?|[ée]d\.?|version)?\s*"
    r"(?P<version_year>\d{4}|\d{2})\s*\)?"
)

TNM_space = r"(?:\s*[,\/]?\s*|\n)"

# Gatekeeper lookahead: a T stage on its own is not enough evidence of a TNM
# mention, since plenty of French clinical text reads that way by accident. It
# must either be qualified (prefix *and* specification) or be followed by an
# N, M or R component.
logic_filter = (
    r"(?="
    # --- BRANCH 1: Standalone T with prefix AND specification ---
    r"(?:" + prefix_pattern + r"T\s*(?:[0-4]|is|[xo])\s*(?:[abcdxm]|mi)\b)"
    r"|"
    # --- BRANCH 2: T followed by N, M, or R ---
    r"(?:(?:" + prefix_pattern + r")?T\s*(?:[0-4]|is|[xo])"
    r"(?:\s*(?:[abcdxm]|mi))?"
    r"(?:\s*\([^()]{1,20}\))?" + TNM_space + r"(?:(?:" + prefix_pattern + r")?"
    r"[NMR]\s*[x0-4+o]))"
    r")"
)

tnm_pattern = (
    r"(?i)"
    r"(?:\b|^)"
    + logic_filter
    + r"(?P<T_component>"
    + tumour_pattern
    + r")"
    # Each optional component is grouped with its preceding TNM_space so that
    # trailing whitespace is not greedily consumed when the component is absent.
    + r"(?:"
    + TNM_space
    + r"(?P<N_component>"
    + node_pattern
    + r"))?"
    + r"(?:"
    + TNM_space
    + r"(?P<M_component>"
    + metastasis_pattern
    + r"))?"
    + r"(?:"
    + TNM_space
    + r"(?P<PL_component>"
    + pleura_pattern
    + r"))?"
    + r"(?:"
    + TNM_space
    + r"(?P<R_component>"
    + resection_pattern
    + r"))?"
    + r"(?:"
    + TNM_space
    + r"(?P<version_component>"
    + version_pattern
    + r"))?"
    + r"(?=[\s\(\)\.,;:/]|$)"
)

# Post-filter applied to matched spans (see `TNMMatcher.process`). These are
# common French clinical abbreviations that read as a valid TNM mention: `atom`
# parses as `aT0M`, `autonom` as `auT0N0m`. The bare `t0`-`t4` are needed too,
# because `o` is a valid stage value: a following word starting with `no`, `mo`
# or `ro` opens `logic_filter`, the component then fails to match and the span
# degrades to the T alone (`t4 nodule` -> `t4`). The lookup lowercases the
# cleaned span text, so these spellings cover every casing.
default_banned_words = [
    "ato",
    "atom",
    "auto",
    "autoa",
    "autonom",
    "ctx",
    "cyto",
    "mto",
    "mtx",
    "mtxd",
    "mtxx",
    "rtx",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "tissunom",
    "to",
    "tox",
]
