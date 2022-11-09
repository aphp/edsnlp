# noinspection SpellCheckingInspection
information = (
    r"(?s)(=====+\s*)?(L\s*e\s*s\sdonnées\s*administratives,\s*sociales\s*|"
    r"I?nfo\s*rmation\s*aux?\s*patients?|"
    r"L[’']AP-HP\s*collecte\s*vos\s*données\s*administratives|"
    r"L[’']Assistance\s*Publique\s*-\s*Hôpitaux\s*de\s*Paris\s*"
    r"\(?AP-HP\)?\s*a\s*créé\s*une\s*base\s*de\s*données)"
    r".{,2000}https?:\/\/recherche\.aphp\.fr\/eds\/droit-opposition[\s\.]*"
)

# Example : NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbW...
bars = r"(?i)([nbw]|_|-|=){5,}"

# Biology tables: Prone to false positive with disease names
biology = r"(\b.*[|¦].*\n)+"

# Leftside note with doctor names
doctors = r"(?mi)(^((dr)|(pr))(\.|\s|of).*)+"

# Mails or websites
web = r"(www\.\S*)|(\S*@\S*)"

# Subsection with ICD-10 Codes
coding = r".*?[a-zA-Z]\d{2,4}.*?(\n|[a-zA-Z]\d{2,4})"

# New page
date = r"\b\d\d/\d\d/\d\d\d\d\b"
ipp = r"80\d{8}"
page = r"((^\d\/\d\s?)|(^\d\d?\/\d\d\?))"
footer = rf"(?i)({page}.*\n?pat.*(ipp)?.*\n?(courrier valid.*)?)"
footer += rf"|(.*{date}.*{ipp}.*)|(imprim.\sle\s{date}.*\d/\d.*\n?pat.*{date})"

pollution = dict(
    information=information,
    bars=bars,
    biology=biology,
    doctors=doctors,
    web=web,
    coding=coding,
    footer=footer,
)
