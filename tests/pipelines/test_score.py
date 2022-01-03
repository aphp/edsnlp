# -*- coding: utf-8 -*-
from typing import List

import spacy

from edsnlp.pipelines.scores import Score
from edsnlp.pipelines.scores.charlson import terms as charlson_terms
from edsnlp.pipelines.scores.sofa import terms as sofa_terms
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
Testons également un autre score.
<ent score_name=SOFA score_value=12 score_method=Maximum>SOFA</ent> maximum : 12.

"""


def test_scores(blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    create_charlson = spacy.registry.get("factories", "charlson")
    create_sofa = spacy.registry.get("factories", "SOFA")

    charlson_default_config = dict(
        regex=charlson_terms.regex,
        after_extract=charlson_terms.after_extract,
        score_normalization=charlson_terms.score_normalization_str,
    )

    sofa_default_config = dict(
        regex=sofa_terms.regex,
        method_regex=sofa_terms.method_regex,
        value_regex=sofa_terms.value_regex,
        score_normalization=sofa_terms.score_normalization_str,
    )

    charlson = create_charlson(
        blank_nlp,
        "charlson",
        **charlson_default_config,
    )
    sofa = create_sofa(
        blank_nlp,
        "SOFA",
        **sofa_default_config,
    )

    def testscore_normalization(raw_score: str):
        if raw_score is not None and int(raw_score) == 0:
            return int(raw_score)

    testscore = Score(
        blank_nlp,
        score_name="TestScore",
        regex=[r"test+score"],
        attr="NORM",
        ignore_excluded=True,
        after_extract=r"(\d+)",
        score_normalization=testscore_normalization,
        window=4,
        verbose=0,
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)
    doc = charlson(doc)
    doc = sofa(doc)
    doc = testscore(doc)

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
        score_normalization=charlson_terms.score_normalization_str,
        window=4,
        verbose=0,
    )
