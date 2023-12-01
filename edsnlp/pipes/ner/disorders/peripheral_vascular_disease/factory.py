from edsnlp import registry

from .patterns import default_patterns
from .peripheral_vascular_disease import PeripheralVascularDiseaseMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="peripheral_vascular_disease",
    span_setter={"ents": True, "peripheral_vascular_disease": True},
)

create_component = registry.factory.register(
    "eds.peripheral_vascular_disease",
    assigns=["doc.ents", "doc.spans"],
)(PeripheralVascularDiseaseMatcher)
