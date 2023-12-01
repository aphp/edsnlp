"""
EDS-NLP
"""

import spacy
from . import patch_spacy_dot_components  # isort: skip
from pathlib import Path

from .evaluate import evaluate
from . import extensions
from .language import *

__version__ = "0.8.0"

BASE_DIR = Path(__file__).parent


for ext in ["Allergie","Action","Certainty","Temporality","Negation","Family"]:
    if not spacy.tokens.Span.has_extension(ext):
        spacy.tokens.Span.set_extension(ext, default=None)

print("Monkey patching spacy.Language.evaluate")
spacy.Language.evaluate = evaluate
