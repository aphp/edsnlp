from edsnlp import registry

from .span_classifier import TrainableSpanClassifier

create_component = registry.factory.register(
    "eds.span_classifier",
    assigns=[],
    deprecated=["eds.span_qualifier"],
)(TrainableSpanClassifier)
