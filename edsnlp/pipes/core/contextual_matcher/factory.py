from edsnlp.core import registry
from edsnlp.pipes.core.contextual_matcher import ContextualMatcher

DEFAULT_CONFIG = dict(
    assign_as_span=False,
    alignment_mode="expand",
    attr="NORM",
    regex_flags=0,
    ignore_excluded=False,
    ignore_space_tokens=False,
    include_assigned=False,
    label_name=None,
    label=None,
    span_setter={"ents": True},
)

create_component = registry.factory.register(
    "eds.contextual-matcher",
    deprecated=["contextual-matcher"],
)(ContextualMatcher)
