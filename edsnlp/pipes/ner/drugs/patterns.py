import json
from typing import Dict, List

from edsnlp import BASE_DIR

drugs_file = BASE_DIR / "resources" / "drugs.json"

def get_patterns(atc: List[str] = None) -> Dict[str, List[str]]:
    with open(drugs_file, "r") as f:
        patterns = json.load(f)
        patterns = {k: v for k, v in patterns.items() if k in atc} if atc else patterns
        return patterns
