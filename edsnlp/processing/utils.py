from typing import Any, Dict, List

from spacy.tokens import Doc


def dummy_extractor(doc: Doc) -> List[Dict[str, Any]]:
    return [
        dict(snippet=ent.text, length=len(ent.text), note_datetime=doc._.note_datetime)
        for ent in doc.ents
    ]
