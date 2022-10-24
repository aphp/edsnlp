from typing import Any, Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher, TerminologyTermMatcher

from . import patterns

PATTERN_CONFIG = dict(
    languages=["FRE"],
    sources=None,
)

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    term_matcher=TerminologyTermMatcher.exact,
    term_matcher_config={},
    pattern_config=PATTERN_CONFIG,
)


@Language.factory(
    "eds.umls", default_config=DEFAULT_CONFIG, assigns=["doc.ents", "doc.spans"]
)
def create_component(
    nlp: Language,
    name: str,
    attr: Union[str, Dict[str, str]],
    ignore_excluded: bool,
    term_matcher: TerminologyTermMatcher,
    term_matcher_config: Dict[str, Any],
    pattern_config: Dict[str, Any],
):

    return TerminologyMatcher(
        nlp,
        label="umls",
        regex=None,
        terms=patterns.get_patterns(pattern_config),
        attr=attr,
        ignore_excluded=ignore_excluded,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
    )
