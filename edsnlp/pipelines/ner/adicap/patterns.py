"""
Source :
https://esante.gouv.fr/sites/default/files/media_entity/documents/cgts_sem_adicap_fiche-detaillee.pdf
"""


d1_4 = r"[A-Z]{4}"
d5_8_v1 = r"\d{4}"
d5_8_v2 = r"\d{4}|[A-Z][0-9A-Z][A-Z][0-9]"
d5_8_v3 = r"[0-9A-Z][0-9][09A-Z][0-9]"
d5_8_v4 = r"0[A-Z][0-9]{2}"


adicap_prefix = r"(?i)(codification|adicap)"
base_code = (
    r"("
    + d1_4
    + r"(?:"
    + d5_8_v1
    + r"|"
    + d5_8_v2
    + r"|"
    + d5_8_v3
    + r"|"
    + d5_8_v4
    + r"))"
)
