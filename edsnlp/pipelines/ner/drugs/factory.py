from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
)


@Language.factory("eds.drugs", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
    ignore_excluded: bool,
):
    return TerminologyMatcher(
        nlp,
        label="drug",
        terms=patterns.get_patterns(),
        regex=dict(),
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
