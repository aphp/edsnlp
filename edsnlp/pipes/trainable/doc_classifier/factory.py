from edsnlp import registry

from .doc_classifier import TrainableDocClassifier

create_component = registry.factory.register(
    "eds.doc_classifier",
    assigns=["doc._.predicted_class"],
    deprecated=[],
)(TrainableDocClassifier)
