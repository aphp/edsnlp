from spacy import Language

from .dementia import DementiaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="dementia",
    span_setter={"ents": True, "dementia": True},
)

create_component = Language.factory(
    "eds.dementia",
    assigns=["doc.ents", "doc.spans"],
)(DementiaMatcher)
