# -*- coding: utf-8 -*-
from typing import List

import spacy

from edsnlp.pipelines.normalizer.normalizer import Normalizer
from edsnlp.pipelines.scores import Score
from edsnlp.pipelines.scores.charlson import terms
from edsnlp.utils.examples import parse_example

example = """
CR-REA.
<ent score_name=charlson score_value=8>Charlson</ent> à l'admission: 8.
- Charlson:
- IMC: 21
Cette phrase teste un score qui s'appelle TestScore.
La seule valeur admissible est 0.
testScore de 1.
<ent score_name=TestScore score_value=0>testtscore</ent> de 0.
"""


def test_scores(blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(
            lowercase=True, accents=True, quotes=True, endlines=False, pollution=False
        ),
    )

    create_charlson = spacy.registry.get("factories", "charlson")

    charlson_default_config = dict(
        regex=terms.regex,
        after_extract=terms.after_extract,
        score_normalization=terms.score_normalization_str,
    )

    charlson = create_charlson(blank_nlp, "charlson", **charlson_default_config)

    def testscore_normalization(raw_score: str):
        if raw_score is not None and int(raw_score) == 0:
            return int(raw_score)

    testscore = Score(
        blank_nlp,
        score_name="TestScore",
        regex=[r"test+score"],
        attr="CUSTOM_NORM",
        after_extract=r"(\d+)",
        score_normalization=testscore_normalization,
        window=4,
        verbose=0,
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)
    doc = testscore(charlson(doc))

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                getattr(ent._, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."


def test_score_factory(blank_nlp):
    factory = spacy.registry.get("factories", "score")
    assert factory(
        blank_nlp,
        "score",
        score_name="TestScore",
        regex=[r"test+score"],
        attr="NORM",
        after_extract=r"(\d+)",
        score_normalization=terms.score_normalization_str,
        window=4,
        verbose=0,
    )
