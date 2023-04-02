from edsnlp.core import registry

from .connective_tissue_disease import ConnectiveTissueDiseaseMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="connective_tissue_disease",
    span_setter={"ents": True, "connective_tissue_disease": True},
)

create_component = registry.factory.register(
    "eds.connective_tissue_disease",
    assigns=["doc.ents", "doc.spans"],
)(ConnectiveTissueDiseaseMatcher)
