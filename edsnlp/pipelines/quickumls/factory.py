from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.quickumls import QuickUMLSComponent

# noinspection PyUnusedLocal
@Language.factory("quickumls")
def create_quickumls_component(
    nlp: Language,
    name: str,
    distribution: str,
):
    return QuickUMLSComponent(nlp, distribution=distribution)
