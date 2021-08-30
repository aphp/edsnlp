from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.quickumls import QuickUMLSComponent

# noinspection PyUnusedLocal
@Language.factory("quickumls")
def create_component(
    nlp: Language,
    name: str,
    distribution: str,
    best_match: bool = True,
    ignore_syntax: bool = False,
):
    return QuickUMLSComponent(
        nlp,
        distribution=distribution,
        best_match=best_match,
        ignore_syntax=ignore_syntax,
    )
