from typing import List

from pytest import fixture, mark

from edsnlp.pipelines import terminations
from edsnlp.pipelines.negation import Negation, terms
from edsnlp.utils.examples import parse_example

negation_examples: List[str] = [
    "Absence d'<ent negated=true>image osseuse d'allure évolutive</ent>.",
    "il n'y a pas de <ent polarity_=NEG>métas,tases</ent>",
    "Le patient n'est pas <ent polarity_=NEG>malade</ent>.",
    "Aucun <ent polarity_=NEG>traitement</ent>.",
    "Le <ent polarity_=AFF>scan</ent> révèle une grosseur.",
    "il y a des <ent polarity_=AFF>métastases</ent>",
    "aucun doute sur les <ent polarity_=AFF>métastases</ent>",
    "il n'y a pas de <ent polarity_=NEG>métastases</ent>",
    "il n'y a pas d' <ent polarity_=NEG>métastases</ent>",
    "il n'y a pas d'<ent polarity_=NEG>métastases</ent>",
    "<ent polarity_=NEG>métas,tases</ent> : non",
    "il n'y a pas d'amélioration de la <ent negated=false>maladie</ent>",
    "Pas de <ent negated=true>lésion pulmonaire avec l'absence de lésion secondaire</ent>.",
]


@fixture
def negation_factory(blank_nlp):

    default_config = dict(
        pseudo=terms.pseudo,
        preceding=terms.preceding,
        following=terms.following,
        termination=terminations.termination,
        verbs=terms.verbs,
        fuzzy=False,
        filter_matches=False,
        attr="LOWER",
        regex=None,
        fuzzy_kwargs=None,
        explain=True,
    )

    def factory(on_ents_only, **kwargs) -> Negation:

        config = dict(**default_config)
        config.update(kwargs)

        return Negation(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("on_ents_only", [True, False])
def test_negation(blank_nlp, negation_factory, on_ents_only):

    negation = negation_factory(on_ents_only=on_ents_only)

    for example in negation_examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = negation(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert bool(ent._.negation_cues) == (modifier.value in {True, "NEG"})

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        if not (getattr(token._, modifier.key) == modifier.value):
                            print()
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
