"""
EDS-NLP
"""
from pathlib import Path

import catalogue
from spacy.util import registry

from . import extensions
from .language import *

from . import patch_spacy_dot_components  # isort: skip


__version__ = "0.7.4"

BASE_DIR = Path(__file__).parent

registry.span_getters = catalogue.create("spacy", "span_getters", entry_points=True)  # type: ignore
registry.annotation_setters = catalogue.create("spacy", "annotation_setters", entry_points=True)  # type: ignore
