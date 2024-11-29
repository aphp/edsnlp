from edsnlp import registry

from .biaffine_dep_parser import TrainableBiaffineDependencyParser

create_component = registry.factory.register(
    "eds.biaffine_dep_parser",
    assigns=["token.head", "token.dep"],
)(TrainableBiaffineDependencyParser)
