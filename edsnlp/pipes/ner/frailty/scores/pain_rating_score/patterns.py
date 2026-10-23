import re

from ..utils import assigned_score_value

pain_cue = re.compile(
    r"algie|br[uû]l|dlr|doul|neuropath|nocicept|radicul",
    flags=re.IGNORECASE,
)
pain_template = re.compile(
    r"[eé]valuation\s+de\s+la\s+douleur|douleurs?\s*[:=-]\s*$|"
    r"v\s+douleurs?\s*[:=-]\s*$",
    flags=re.IGNORECASE,
)
simple_en_score = re.compile(
    r"^\s*en(?:\s+(?:a|à|au|aux|de|du|des|est|etait|était|"
    r"cotee?|cotée?|evaluee?|évaluée?|notee?|notée?|chiffree?|chiffrée?|"
    r"actuel(?:le)?|habituel(?:le)?|ce|crise|effort|geste|jour|marche|matin|"
    r"max(?:imale?|imum)?|mobilisation|moyenne|nocturne|nuit|pendant|pire|"
    r"procedure|repos|soir))*"
    r"\s*(?:[:=]|[<>]=?)?\s*"
    r"\d+(?:[,.]\d+)?(?:\s*/\s*(?:10|100))?\s*\.?\s*$",
    flags=re.IGNORECASE,
)
blacklisted_score_text = re.compile(
    r"(?:^\s*en\s*(?:[-–]\s*)?cours\b)|☎",
    flags=re.IGNORECASE,
)
blacklisted_eva_context = re.compile(
    r"(?:fatigue\s+muscul\w*(?:\s*\()?|dyspn[eé]e|prurit)\s*(?:[:=-]\s*)?$",
    flags=re.IGNORECASE,
)
blacklisted_score_words = {
    "activite",
    "apnee",
    "asthenie",
    "dyspnee",
    "efficacite",
    "fatigue",
    "gene",
    "globale",
    "maladie",
    "patient",
    "prurit",
    "raideur",
    "secheresse",
}
blacklisted_score_word_prefixes = {
    "aliment",
    "consomm",
    "ophtalmo",
}
allowed_uppercase_score_words = {
    "AU",
    "DOULEUR",
    "DOULEURS",
    "EN",
    "EVA",
    "EVENDOL",
    "EVN",
    "OU",
    "REPOS",
}

default_patterns = [
    dict(
        source="pain_rating_score",
        terms=["EN", "EVA", "EVN"],
        assign=[
            dict(
                name="value",
                regex=assigned_score_value,
                window=(0, 7),
                reduce_mode="keep_first",
            ),
        ],
    )
]
