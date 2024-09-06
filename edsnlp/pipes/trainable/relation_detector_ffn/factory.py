from edsnlp import registry

from .relation_detector_ffn import RelationDetectorFFN

create_component = registry.factory.register(
    "eds.relation_detector_ffn",
    assigns=[],
    deprecated=[],
)(RelationDetectorFFN)
