from spacy import Language

from .liver_disease import LiverDiseaseMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="liver_disease",
    span_setter={"ents": True, "liver_disease": True},
)

create_component = Language.factory(
    "eds.liver_disease",
    assigns=["doc.ents", "doc.spans"],
)(LiverDiseaseMatcher)
