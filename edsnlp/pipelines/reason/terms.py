from typing import List

reasons = dict(
    reasons=[
        "(?i)motif de l.?hospitalisation : .+",
        "(?i)hospitalis[ée].?.*(pour|. cause|suite [àa]).+",
        "(?i)(consulte|prise en charge(?!\set\svous\sassurer\sun\straitement\sadapté)).*pour.+",
        "(?i)motif\sd.hospitalisation\s:.+",
        "(?i)au total\s?\:?\s?\n?.+",
    ]
)
