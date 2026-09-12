# noinspection SpellCheckingInspection
information = [
    (
        # Avoid retrying the optional separator from inside the same equals run
        r"(?s)((?<![=])=====+\s*)?(L\s*e\s*s\sdonnées\s*administratives,\s*sociales\s*|"
        r"I?nfo\s*rmation\s*aux?\s*patients?|"
        r"L[’']AP-HP\s*collecte\s*vos\s*données\s*administratives|"
        r"L[’']Assistance\s*Publique\s*-\s*Hôpitaux\s*de\s*Paris\s*"
        r"\(?AP-HP\)?\s*a\s*créé\s*une\s*base\s*de\s*données)"
        r".{,2000}https?:\/\/recherche\.aphp\.fr\/eds\/droit-opposition[\s\.]*"
    ),
    (
        r"(?si)l’arrêt\s*du\s*tabac\s*permet\s*de\s*diminuer\s*le\s*risque\s*"
        r"de\s*maladie\s*cardiovasculaire."
    ),
]
# Example : NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbW...
bars = r"(?i)([nbw]|_|-|=){5,}"

# Biology tables: Prone to false positive with disease names
# Word characters include superscript numbers and exclude combining marks
word = r"[\p{L}\p{N}_]"
# Scan each line once and exclude leading punctuation from the pollution span
biology = rf"^[^\p{{L}}\p{{N}}_\n]*\K({word}[^\n|¦]*[|¦][^\n]*\n)+"

# Leftside note with doctor names
doctors = r"(?mi)(^((dr)|(pr))(\.|\s|of).*)+"

# Mails or websites
web = [
    r"(www\.\S*)",
    r"(?<!\S)(\S*@\S*)",
    r"(?<!\S)\S*\.(?:fr|com|net|org)",
]

# Subsection with ICD-10 Codes
# Resume at the previous match end for multiple coding sections on one line
coding = r"(?:^|\G).*? \(\d+\) [a-zA-Z]\d{2,4}.*?(\n|[a-zA-Z]\d{2,4})"


# New page
date = rf"(?<!{word})\d\d/\d\d/\d\d\d\d(?!{word})"
ipp = r"80\d{8}"
# Notice separators include Unicode whitespace and control separators
space = r"[\s\x1c-\x1f]"
page = rf"((^\d\/\d{space}?)|(^\d\d?\/\d\d\?))"
# Footer words accept dotless i as a case variant
footer = rf"(?i)({page}.*\n?pat.*\n?(courr[iı]er val[iı]d.*)?)"
# The first date is sufficient to find an IPP later on the same line
footer += rf"|((?:^|\G)(?>.*?{date}).*{ipp}.*)"
# Commit to the first page number and prefer a patient line after the newline
footer += (
    rf"|((?:^|\G)(?>.*?\K[iı]mpr[iı]m.{space}le{space}{date})"
    rf"(?>.*?\d/\d)(?:.*\npat.*{date}|(?>.*?pat).*{date}))"
)

# Word split in the middle due to line break
intraword_split = r"(?<![\W\d_])-\n"

pollution = dict(
    information=information,
    bars=bars,
    biology=biology,
    doctors=doctors,
    web=web,
    coding=coding,
    footer=footer,
    intraword_split=intraword_split,
)

default_enabled = dict(
    information=True,
    bars=True,
    biology=False,
    doctors=True,
    web=True,
    coding=False,
    footer=True,
    intraword_split=True,
)
