"""
EDS-NLP
"""

from pathlib import Path
from spacy import pipeline as _spacy_pipeline  # noqa: F401
from . import extensions
from . import patch_spacy
from .core.pipeline import Pipeline, blank, load
from .core.registry import registry
# from . import language
import edsnlp.data  # noqa: F401

__version__ = "0.10.0beta1"

BASE_DIR = Path(__file__).parent
