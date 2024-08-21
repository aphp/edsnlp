import json
from typing import Dict, List

from edsnlp import BASE_DIR

drugs_file = BASE_DIR / "resources" / "drugs.json"


def filter_dict_by_keys(D: Dict[str, List[str]], L: List[str]):
    filtered_dict = {
        k: v for k, v in D.items() if any(k.startswith(prefix) for prefix in L)
    }
    return filtered_dict


def get_patterns(atc: List[str] = None) -> Dict[str, List[str]]:
    with open(drugs_file, "r") as f:
        patterns = json.load(f)
        patterns = {k: v for k, v in patterns.items() if k in atc} if atc else patterns
        return patterns
