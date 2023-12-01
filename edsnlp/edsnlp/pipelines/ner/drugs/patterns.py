import json
from typing import Dict, List

from edsnlp import BASE_DIR

drugs_file = BASE_DIR / "resources" / "drugs.json"


def get_patterns() -> Dict[str, List[str]]:

    with (open(drugs_file, "r")) as f:
        return json.load(f)
