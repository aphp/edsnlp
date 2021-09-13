from typing import List
from edsnlp.utils.examples import parse_example

from edsnlp.pipelines.negation import terms, Negation
from edsnlp.pipelines import terminations

from pytest import fixture

negation_examples: List[str] = [
    "Le patient n'est pas <ent polarity_=NEG>malade</ent>.",
    "Aucun <ent polarity_=NEG>traitement</ent>.",
    "Le <ent polarity_=AFF>scan</ent> révèle une grosseur.",
    "il y a des <ent polarity_=AFF>métastases</ent>",
    "il n'y a pas de <ent polarity_=NEG>métastases</ent>",
    "il n'y a pas d' <ent polarity_=NEG>métastases</ent>",
    "il n'y a pas d'<ent polarity_=NEG>métastases</ent>",
    "il n'y a pas de <ent polarity_=NEG>métas,tases</ent>",
    "<ent polarity_=NEG>métas,tases</ent> : non",
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
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return Negation(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


def test_negation_no_ents_only(blank_nlp, negation_factory):

    negation = negation_factory(on_ents_only=False)

    for example in negation_examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [doc.char_span(ent.start_char, ent.end_char) for ent in entities]

        doc = negation(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."


def test_negation_ents_only(blank_nlp, negation_factory):

    negation = negation_factory(on_ents_only=True)

    for example in negation_examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [doc.char_span(ent.start_char, ent.end_char) for ent in entities]

        doc = negation(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."
