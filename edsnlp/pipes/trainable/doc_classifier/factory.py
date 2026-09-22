from edsnlp import registry

from .doc_classifier import TrainableDocClassifier

# The extensions written by this component are named after its heads and are
# therefore only known at instantiation time, hence the empty `assigns`.
create_component = registry.factory.register(
    "eds.doc_classifier",
    assigns=[],
    deprecated=[],
)(TrainableDocClassifier)
