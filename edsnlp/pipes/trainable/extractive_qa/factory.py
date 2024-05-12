from typing import TYPE_CHECKING

from edsnlp import registry

from .extractive_qa import TrainableExtractiveQA

create_component = registry.factory.register(
    "eds.extractive_qa",
    assigns=[],
    deprecated=[],
)(TrainableExtractiveQA)

if TYPE_CHECKING:
    create_component = TrainableExtractiveQA
