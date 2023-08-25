from typing import Any, Dict, Iterable

from spacy.training import Example

from edsnlp import registry


def speed_scorer(
    examples: Iterable[Example], duration: float, cfg=None
) -> Dict[str, Any]:
    words_count = [len(eg.predicted) for eg in examples]
    num_words = sum(words_count)
    num_docs = len(words_count)

    return {
        "wps": num_words / duration,
        "dps": num_docs / duration,
    }


@registry.scorers.register("speed")
def create_speed_scorer():
    return speed_scorer
