from edsnlp import registry

from .multi_query_qa import TrainableMultiQueryQA

create_component = registry.factory.register(
    "eds.multi_query_qa",
    assigns=["doc.spans"],
)(TrainableMultiQueryQA)
