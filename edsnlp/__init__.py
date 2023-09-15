"""
EDS-NLP
"""

from . import patch_spacy_dot_components  # isort: skip
from pathlib import Path

from . import extensions
from .language import *

__version__ = "0.9.0"

BASE_DIR = Path(__file__).parent
