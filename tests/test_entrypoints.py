import catalogue
import pytest

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points

try:
    import torch.nn
except ImportError:
    torch = None

if torch is None:
    pytest.skip("torch not installed", allow_module_level=True)


def test_entrypoints():
    ep = entry_points()
    namespaces = ep.groups if hasattr(ep, "groups") else ep.keys()
    for ns in namespaces:
        if ns.startswith("spacy_") or ns.startswith("edsnlp_"):
            reg = catalogue.Registry(ns.split("_"), entry_points=True)
            reg.get_all()
