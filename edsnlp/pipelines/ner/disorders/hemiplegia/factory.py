from spacy import Language

from .hemiplegia import HemiplegiaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="hemiplegia",
    span_setter={"ents": True, "hemiplegia": True},
)

create_component = Language.factory(
    "eds.hemiplegia",
    assigns=["doc.ents", "doc.spans"],
)(HemiplegiaMatcher)
