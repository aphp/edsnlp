from spacy.language import Language

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.deprecation import deprecated_factory

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

create_component = deprecated_factory(
    "contextual-matcher",
    "eds.contextual-matcher",
)(ContextualMatcher)
create_component = Language.factory(
    "eds.contextual-matcher",
)(create_component)
