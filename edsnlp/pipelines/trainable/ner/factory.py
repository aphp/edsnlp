from edsnlp import registry

from .ner import TrainableNER

create_component = registry.factory.register(
    "eds.ner_crf",
    assigns=["doc.ents", "doc.spans"],
    deprecated=[
        "eds.nested_ner",
        "nested_ner",
    ],
)(TrainableNER)
