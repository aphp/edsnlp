from edsnlp.core import registry

from .incontinence import IncontinenceMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="incontinence",
    label="incontinence",
    span_setter={"ents": True, "incontinence": True},
)

create_component = registry.factory.register(
    "eds.incontinence",
    assigns=["doc.ents", "doc.spans"],
)(IncontinenceMatcher)
