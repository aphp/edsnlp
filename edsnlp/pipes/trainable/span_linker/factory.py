from typing import TYPE_CHECKING

from edsnlp import registry

from .span_linker import TrainableSpanLinker

create_component = registry.factory.register(
    "eds.span_linker",
    assigns=[],
    deprecated=[],
)(TrainableSpanLinker)

if TYPE_CHECKING:
    create_component = TrainableSpanLinker
