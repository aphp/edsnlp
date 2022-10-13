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
doctors = r"(?mi)(^((dr)|(pr)).*)+"

# Mails or websites
web = r"(www\.\S*)|(\S*@\S*)"

# Subsection with ICD-10 Codes
coding = r".*?[a-zA-Z]\d{2,4}.*?(\n|[a-zA-Z]\d{2,4})"

# New page
footer = r"(?i)^\d\/\d\s?pat.*ipp.*"

pollution = dict(
    information=information,
    bars=bars,
    biology=biology,
    doctors=doctors,
    web=web,
    coding=coding,
    footer=footer,
)
