from edsnlp.core import registry
from edsnlp.pipes.misc.tables import TablesMatcher

DEFAULT_CONFIG = dict(
    tables_pattern=None,
    sep_pattern=None,
    attr="TEXT",
    ignore_excluded=True,
)

create_component = registry.factory.register(
    "eds.tables",
    assigns=["doc.spans", "doc.ents"],
    deprecated=["tables"],
)(TablesMatcher)
