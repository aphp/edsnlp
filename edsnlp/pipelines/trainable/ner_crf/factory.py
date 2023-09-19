from edsnlp import registry

from .ner_crf import TrainableNerCrf

create_component = registry.factory.register(
    "eds.ner_crf",
    assigns=["doc.ents", "doc.spans"],
    deprecated=[
        "eds.nested_ner",
        "nested_ner",
    ],
)(TrainableNerCrf)
