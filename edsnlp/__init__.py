"""
EDS-NLP
"""

# fmt: off
from . import patch_spacy  # noqa: F401
from pathlib import Path

from . import extensions  # noqa: F401
from .language import *
# fmt: on

__version__ = "0.9.1"

BASE_DIR = Path(__file__).parent
