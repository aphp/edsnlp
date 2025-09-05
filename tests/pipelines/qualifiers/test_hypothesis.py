from typing import List

from pytest import fixture, mark

from edsnlp.pipelines.qualifiers.hypothesis import Hypothesis
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    "Possible <ent hypothesis_=HYP>covid-19</ent>",
    (
        "Plusieurs <ent hypothesis_=HYP>diagnostics</ent> sont envisagés. "
        "Le patient est informé."
    ),
    "même si <ent hypothesis=False>le patient est jeune</ent>.",
    "Suspicion de <ent hypothesis_=HYP>diabète</ent>.",
    "Le ligament est <ent hypothesis_=CERT>rompu</ent>.",
    "Probablement du diabète mais pas de <ent hypothesis_=CERT>cécité</ent>.",
]


@fixture
def hypothesis_factory(blank_nlp):

    default_config = dict(
        pseudo=None,
        preceding=None,
        following=None,
        termination=None,
        verbs_hyp=None,
        verbs_eds=None,
        attr="NORM",
        within_ents=False,
        explain=True,
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return Hypothesis(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("on_ents_only", [True, False])
def test_hypothesis(blank_nlp, hypothesis_factory, on_ents_only):

    hypothesis = hypothesis_factory(on_ents_only)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = hypothesis(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert bool(ent._.hypothesis_cues) == (modifier.value in {"HYP", True})

                assert getattr(ent._, modifier.key) == modifier.value

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."


def test_hypothesis_within_ents(blank_nlp, hypothesis_factory):

    hypothesis = hypothesis_factory(True, within_ents=True)

    examples = [
        "<ent hypothesis_=HYP>Diabète, probablement de type 2</ent>.",
    ]

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = hypothesis(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert bool(ent._.hypothesis_cues) == (modifier.value in {"HYP", True})
