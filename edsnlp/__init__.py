from pathlib import Path

from . import extensions

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("edsnlp")
except ImportError:
    __version__ = "dev"

BASE_DIR = Path(__file__).parent
