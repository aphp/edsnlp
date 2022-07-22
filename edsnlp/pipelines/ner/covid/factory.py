from typing import Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="LOWER",
    ignore_excluded=False,
)


@Language.factory(
    "eds.covid",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    attr: Union[str, Dict[str, str]],
    ignore_excluded: bool,
):

    return GenericMatcher(
        nlp,
        terms=None,
        regex=dict(covid=patterns.pattern),
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
