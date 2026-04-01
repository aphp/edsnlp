from edsnlp.core import registry

from .patterns import default_patterns
from .polypharmacy_status import PolypharmacyStatusMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="polypharmacy_status",
    label="polypharmacy_status",
    span_setter={"ents": True, "polypharmacy_status": True},
)

create_component = registry.factory.register(
    "eds.polypharmacy_status",
    assigns=["doc.ents", "doc.spans"],
)(PolypharmacyStatusMatcher)
