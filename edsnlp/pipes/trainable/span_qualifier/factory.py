from edsnlp import registry

from .span_qualifier import TrainableSpanQualifier, TrainableSpanQualifierRoberta

create_component = registry.factory.register(
    "eds.span_qualifier",
    assigns=[],
    deprecated=[],
)(TrainableSpanQualifier)

create_component2 = registry.factory.register(
    "eds.span_qualifier_roberta",
    assigns=[],
    deprecated=[],
)(TrainableSpanQualifierRoberta)
