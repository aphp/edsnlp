from typing import Any, Dict

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher, TerminologyTermMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    term_matcher=TerminologyTermMatcher.exact,
    term_matcher_config={},
)


@Language.factory(
    "eds.drugs",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
    ignore_excluded: bool,
    term_matcher: TerminologyTermMatcher,
    term_matcher_config: Dict[str, Any],
):
    return TerminologyMatcher(
        nlp,
        label="drug",
        terms=patterns.get_patterns(),
        regex=dict(),
        attr=attr,
        ignore_excluded=ignore_excluded,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
