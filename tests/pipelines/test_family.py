from typing import List
from edsnlp.utils.examples import parse_example

from pytest import fixture, mark

from edsnlp.pipelines.family import FamilyContext, terms

examples: List[str] = [
    "Le père du patient a eu un <ent family_=FAMILY>cancer du colon</ent>. La mère se porte bien.",
    "Antécédents familiaux : <ent family_=FAMILY>diabète</ent>.",
    "Un <ent family_=PATIENT>relevé</ent> sanguin a été effectué.",
]


@fixture
def family_factory(blank_nlp):

    default_config = dict(
        family=terms.family,
        fuzzy=False,
        filter_matches=False,
        attr="LOWER",
        explain=True,
        regex=None,
        fuzzy_kwargs=None,
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return FamilyContext(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("on_ents_only", [True, False])
@mark.parametrize("use_sections", [True, False])
def test_family(blank_nlp, family_factory, on_ents_only, use_sections):

    family = family_factory(on_ents_only=on_ents_only, use_sections=use_sections)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = family(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert bool(ent._.family_cues) == (modifier.value in {"FAMILY", True})

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
