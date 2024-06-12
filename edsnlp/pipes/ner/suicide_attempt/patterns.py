patterns = {
    "suicide_attempt_unspecific": [
        r"\b(?<!\.)(?<!Voie\s\d\s\:\s)(?<!Voie\sd.abord\s\:\s)(?<!surface\s)(?<!d[ée]sorientation\s)(?<!abord\s)(?<!ECG\s:\s)(?<!volume\s)(?<!\d\s[mc]m\sde\sla\s)(?<!\d[mc]m\sde\sla\s)(?<!au\scontact\sde\sla\s)T\.?S\.?(?![\.A-Za-z])(?!\sapyr[eé]tique)(?!.+TRANSSEPTAL)(?!.+T[34])(?!.+en\sr.gression)\b",
        r"(?<!\.)T\.S\.(?![A-Za-z])",
        r"\b(?<!.)TS\.\B",
        r"(?i)tentative[s]?\s+de\s+sui?cide",
        r"(?i)tent[ée]\s+de\s+((se\s+(suicider|tuer))|(mettre\s+fin\s+[àa]\s+((ses\s+jours?)|(sa\s+vie))))",
    ],
    "autolysis": [r"(?i)tentative\s+d'autolyse", r"(?i)autolyse"],
    "intentional_drug_overdose": [
        r"(?i)(intoxication|ingestion)\s+m[ée]dicamenteuse\s+volontaire",
        r"(?i)\b(i\.?m\.?v\.?)\b",
        r"(?i)(intoxication|ingestion)\s*([a-zA-Z0-9_éàèôê\-]+\s*){0,3}\s*volontaire",
        r"TS\s+med\s+polymedicamenteuse",
        r"TS\s+(poly)?([\s-])?m[ée]dicamenteuse",
    ],
    "jumping_from_height": [
        r"(?i)tentative[s]?\s+de\s+d[ée]fenestration",
        r"(?i)(?<!id[ée]es?\sde\s)d[ée]fenestration(?!\saccidentelle)",
        r"(?i)d[ée]fenestration\s+volontaire",
        r"(?i)d[ée]fenestration\s+intentionnelle",
        r"(?i)jet.r?\sd.un\spont",
    ],
    "cuts": [r"(?i)phl[ée]botomie"],
    "strangling": [r"(?i)pendaison"],
    "self_destructive_behavior": [r"(?i)autodestruction"],
    "burn_gas_caustic": [r"(?i)ing[eé]stion\sde\s(produit\s)?caustique"],
}
