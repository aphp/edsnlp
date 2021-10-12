import sys
from os.path import abspath, dirname, join

REPO_PATH = abspath(join(dirname(__file__), ".."))
sys.path.insert(0, REPO_PATH)
