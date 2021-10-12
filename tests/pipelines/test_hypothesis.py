from typing import List

from pytest import fixture, mark

from edsnlp.pipelines import terminations
from edsnlp.pipelines.hypothesis import Hypothesis, terms
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    "Plusieurs <ent hypothesis_=HYP>diagnostics</ent> sont envisagés. Le patient est informé.",
    "même si <ent hypothesis=False>le patient est jeune</ent>.",
    "Suspicion de <ent hypothesis_=HYP>diabète</ent>.",
    "Le ligament est <ent hypothesis_=CERT>rompu</ent>.",
    "Probablement du diabète mais pas de <ent hypothesis_=CERT>cécité</ent>.",
    "<ent hypothesis_=HYP>Tabac</ent> :\n",
]


@fixture
def hypothesis_factory(blank_nlp):

    default_config = dict(
        pseudo=terms.pseudo,
        confirmation=terms.confirmation,
        preceding=terms.preceding,
        following=terms.following,
        termination=terminations.termination,
        verbs_hyp=terms.verbs_hyp,
        verbs_eds=terms.verbs_eds,
        fuzzy=False,
        filter_matches=True,
        attr="LOWER",
        explain=True,
        regex=None,
        fuzzy_kwargs=None,
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

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
