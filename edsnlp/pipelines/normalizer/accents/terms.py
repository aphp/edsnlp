from typing import List, Tuple

# Accentuated characters
accents: List[Tuple[str, str]] = [
    ("ç", "c"),
    ("àáâä", "a"),
    ("èéêë", "e"),
    ("ìíîï", "i"),
    ("òóôö", "o"),
    ("ùúûü", "u"),
]
# Add uppercase
accents += [(k.upper(), v.upper()) for k, v in accents]
