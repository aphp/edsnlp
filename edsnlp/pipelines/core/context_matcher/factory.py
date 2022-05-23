from spacy.language import Language

from edsnlp.pipelines.core.context_matcher import ContextMatcher, MatcherType

DEFAULT_CONFIG = dict(
    matcher=MatcherType.phrase,
    attr="NORM",
    ignore_excluded=False,
)


@Language.factory(
    "eds.context-matcher",
    default_config=DEFAULT_CONFIG,
)
def create_component(
    nlp: Language,
    name: str,
    matcher: MatcherType,
    attr: str,
    ignore_excluded: bool,
):
    return ContextMatcher(
        nlp,
        matcher=matcher,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
