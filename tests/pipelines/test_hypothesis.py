from typing import List
from edsnlp.utils.examples import parse_example

from edsnlp.pipelines.hypothesis import Hypothesis, terms

from pytest import fixture, mark


examples: List[str] = [
    "Plusieurs <ent hypothesis_=HYP>diagnostics</ent> sont envisagés.",
    "Suspicion de <ent hypothesis_=HYP>diabète</ent>.",
    "Le ligament est <ent hypothesis_=CERT>rompu</ent>.",
]


@fixture
def hypothesis_factory(blank_nlp):

    default_config = dict(
        pseudo=terms.pseudo,
        confirmation=terms.confirmation,
        preceding=terms.preceding,
        following=terms.following,
        verbs_hyp=terms.verbs_hyp,
        verbs_eds=terms.verbs_eds,
        fuzzy=False,
        filter_matches=True,
        attr="LOWER",
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
        doc.ents = [doc.char_span(ent.start_char, ent.end_char) for ent in entities]

        doc = hypothesis(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."
