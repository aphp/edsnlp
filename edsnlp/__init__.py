"""
EDS-NLP
"""

import sys
import importlib.abc
import importlib.util
from pathlib import Path
from spacy import pipeline as _spacy_pipeline  # noqa: F401
from . import extensions
from . import patch_spacy
from .core.pipeline import Pipeline, blank, load
from .core.registries import registry
import edsnlp.data  # noqa: F401
import edsnlp.pipes
from . import reducers

__version__ = "0.16.0"

BASE_DIR = Path(__file__).parent


# Everything below is to support deprecated use of edsnlp.pipelines
# route imports of submodules of edsnlp.pipelines to their edsnlp.pipes counterparts.
# The same is done for edsnlp.scorers -> edsnlp.metrics

class AliasPathFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # pragma: no cover
        if not fullname.startswith("edsnlp."):
            return None
        if fullname.startswith("edsnlp.pipelines"):
            new_name = "edsnlp.pipes" + fullname[16:]
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if fullname.startswith("edsnlp.core.lazy_collection"):
            new_name = "edsnlp.core.stream" + fullname[27:]
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if fullname.startswith("edsnlp.optimization"):
            new_name = "edsnlp.training.optimizer" + fullname[19:]
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if fullname.startswith("edsnlp.scorers"):
            new_name = "edsnlp.metrics" + fullname[14:]
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if fullname.startswith("edsnlp.metrics.span_classification"):
            new_name = "edsnlp.metrics.span_attributes" + fullname[34:]
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if "span_qualifier" in fullname.split("."):
            new_name = fullname.replace("span_qualifier", "span_classifier")
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec
        if "measurements" in fullname.split("."):
            new_name = fullname.replace("measurements", "quantities")
            spec = importlib.util.spec_from_loader(fullname, AliasLoader(new_name))
            return spec


class AliasLoader(importlib.abc.Loader):
    def __init__(self, alias):
        self.alias = alias

    def create_module(self, spec):
        # import the edsnlp.pipe.* module first, then alias it to edsnlp.pipelines.*
        module = importlib.import_module(self.alias)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


# Add the custom finder to sys.path_hooks
sys.meta_path.insert(0, AliasPathFinder())
