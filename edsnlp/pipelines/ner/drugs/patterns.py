import json

from edsnlp import BASE_DIR

drugs_file = BASE_DIR / "resources" / "drugs.json"

with (open(drugs_file, "r")) as f:
    terms = json.load(f)
