from spacy import Language

from .lymphoma import LymphomaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="lymphoma",
    span_setter={"ents": True, "lymphoma": True},
)

create_component = Language.factory(
    "eds.lymphoma",
    assigns=["doc.ents", "doc.spans"],
)(LymphomaMatcher)
