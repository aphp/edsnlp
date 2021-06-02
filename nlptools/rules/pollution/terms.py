# Droits et information des patients
informations = (
    r"(?s)(=====+\s*)?(L\s*e\s*s\sdonnées\s*administratives,\s*sociales\s*|"
    r"I?nfo\s*rmation\s*aux?\s*patients?|"
    r"L’AP-HP\s*collecte\s*vos\s*données\s*administratives|"
    r"L’Assistance\s*Publique\s*-\s*Hôpitaux\s*de\s*Paris\s*\(?AP-HP\)?\s*a\s*créé\s*une\s*base\s*de\s*données)"
    r".{,2000}https?:\/\/recherche\.aphp\.fr\/eds\/droit-opposition[\s\.]*"
)

# Exemple : NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbW...
bars = r"(?i)\b([nbw]{5,}|[_\-]{5,})\b"


pollutions = dict(
    informations=informations,
    bars=bars,
)
