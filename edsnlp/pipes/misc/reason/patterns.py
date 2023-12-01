reasons = dict(
    reasons=[
        r"(?i)motif de l.?hospitalisation : .+",
        r"(?i)hospitalis[ée].?.*(pour|. cause|suite [àa]).+",
        (
            r"(?i)(consulte|prise en charge"
            r"(?!\set\svous\sassurer\sun\straitement\sadapté)).*pour.+"
        ),
        r"(?i)motif\sd.hospitalisation\s:.+",
        r"(?i)au total\s?\:?\s?\n?.+",
        r"(?i)motif\sde\sla\sconsultation",
        r"(?i)motif\sd.admission",
        r"(?i)conclusion\smedicale",
    ]
)

sections_reason = ["motif", "conclusion"]

section_exclude = [
    "antécédents",
    "antécédents familiaux",
    "histoire de la maladie",
]
