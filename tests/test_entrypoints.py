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


def test_readers_and_writers_entrypoints():
    # Map of expected entry points for readers and writers
    expected_readers = {
        "spark": "from_spark",
        "pandas": "from_pandas",
        "json": "read_json",
        "parquet": "read_parquet",
        "standoff": "read_standoff",
        "brat": "read_brat",
        "conll": "read_conll",
        "polars": "from_polars",
    }
    expected_writers = {
        "spark": "to_spark",
        "pandas": "to_pandas",
        "json": "write_json",
        "standoff": "write_standoff",
        "brat": "write_brat",
        "polars": "to_polars",
        "parquet": "write_parquet",
    }
    eps = entry_points()
    if hasattr(eps, "select"):
        readers_eps = eps.select(group="edsnlp_readers")
        writers_eps = eps.select(group="edsnlp_writers")
    else:
        readers_eps = eps.get("edsnlp_readers", [])
        writers_eps = eps.get("edsnlp_writers", [])

    readers = {ep.name for ep in readers_eps}
    writers = {ep.name for ep in writers_eps}
    for name in expected_readers:
        assert name in readers, f"Reader entry point '{name}' is missing"
    for name in expected_writers:
        assert name in writers, f"Writer entry point '{name}' is missing"
