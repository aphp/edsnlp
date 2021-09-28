from typing import List
from edsnlp.utils.examples import parse_example

from pytest import fixture, mark

from edsnlp.pipelines.antecedents import Antecedents, terms
from edsnlp.pipelines import terminations

examples: List[str] = [
    "Antécédents d'<ent antecedent_=ATCD>AVC</ent>.",
    "atcd <ent antecedent_=ATCD>chirurgicaux</ent> : aucun.",
    "Le patient est <ent antecedent_=CURRENT>fumeur</ent>.",
    # Les sections ne sont pas utilisées par défaut
    "\nv Antecedents :\n- <ent antecedent_=CURRENT>appendicite</ent>\nv Motif :\n<ent antecedent_=CURRENT>malaise</ent>",
]


@fixture
def antecedents_factory(blank_nlp):

    default_config = dict(
        antecedents=terms.antecedents,
        termination=terminations.termination,
        use_sections=False,
        fuzzy=False,
        filter_matches=False,
        attr="LOWER",
        regex=None,
        fuzzy_kwargs=None,
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return Antecedents(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("use_sections", [True, False])
@mark.parametrize("on_ents_only", [True, False])
def test_antecedents(blank_nlp, antecedents_factory, on_ents_only, use_sections):

    antecedents = antecedents_factory(on_ents_only, use_sections=use_sections)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = antecedents(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
