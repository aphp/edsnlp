from edsnlp import registry

from .span_qualifier import TrainableSpanQualifier

create_component = registry.factory.register(
    "eds.span_qualifier",
    assigns=[],
    deprecated=[],
)(TrainableSpanQualifier)
