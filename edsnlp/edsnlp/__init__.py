"""
EDS-NLP
"""

from pathlib import Path

import spacy

from . import extensions
from .evaluate import evaluate
from .language import *

from . import patch_spacy_dot_components  # isort: skip


__version__ = "0.8.0"

BASE_DIR = Path(__file__).parent


for ext in ["Allergie", "Action", "Certainty", "Temporality", "Negation", "Family"]:
    if not spacy.tokens.Span.has_extension(ext):
        spacy.tokens.Span.set_extension(ext, default=None)

print("Monkey patching spacy.Language.evaluate")
spacy.Language.evaluate = evaluate
