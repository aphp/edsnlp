# -*- coding: utf-8 -*-
from typing import List
from edsnlp.utils.examples import parse_example

from edsnlp.pipelines.normalizer.normalizer import Normalizer
from edsnlp.pipelines.advanced import AdvancedRegex

example = """
Faible fracture du pied.
Le patient présente plusieurs <ent label_=fracture>fêlures</ent>.
Présence de fractures de fatigues.
"""


def test_advanced(blank_nlp):

    normalizer = Normalizer(
        lowercase=True,
        remove_accents=True,
        normalize_quotes=True,
    )

    regex_config = dict(
        fracture=dict(
            regex=[r"fracture", r"felure"],
            attr="NORM",
            before_exclude="petite|faible",
            after_exclude="legere|de fatigue",
        )
    )

    advanced_regex = AdvancedRegex(
        nlp=blank_nlp,
        regex_config=regex_config,
        window=5,
        verbose=True,
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)

    doc = advanced_regex(normalizer(doc))

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                getattr(ent, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."
