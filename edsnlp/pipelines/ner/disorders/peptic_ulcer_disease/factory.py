from spacy import Language

from .patterns import default_patterns
from .peptic_ulcer_disease import PepticUlcerDiseaseMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="peptic_ulcer_disease",
    span_setter={"ents": True, "peptic_ulcer_disease": True},
)

create_component = Language.factory(
    "eds.peptic_ulcer_disease",
    assigns=["doc.ents", "doc.spans"],
)(PepticUlcerDiseaseMatcher)
