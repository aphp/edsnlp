from edsnlp.core import registry

from .incontinence_status import IncontinenceMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="incontinence_status",
    label="incontinence_status",
    span_setter={"ents": True, "incontinence_status": True},
)

create_component = registry.factory.register(
    "eds.incontinence_status",
    assigns=["doc.ents", "doc.spans"],
)(IncontinenceMatcher)
