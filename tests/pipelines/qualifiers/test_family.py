from typing import List

from pytest import fixture, mark
from spacy.tokens import Span

from edsnlp.pipes.qualifiers.family import FamilyContext
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    (
        "Le père est <ent family=True>asthmatique</ent>, "
        "sans traitement traitement particulier."
    ),
    "Son père est atteint de la <ent family=True>COVID</ent>",
    "Son père a une infection au <ent family=True>COVID</ent>",
    "Son père a une possible infection au <ent family=True>COVID</ent>",
    (
        "Le père du patient a eu un <ent family_=FAMILY>cancer du colon</ent>. "
        "La mère se porte bien."
    ),
    "Antécédents familiaux : <ent family_=FAMILY>diabète</ent>.",
    "Un <ent family_=PATIENT>relevé</ent> sanguin a été effectué.",
    (
        "Antécédent familiaux de diabète mais pas "
        "<ent family_=PATIENT>détecté</ent> jusqu'ici."
    ),
    "mère : <ent family=True>diabète de type II</ent>",
]


@fixture
def family_factory(blank_nlp):
    default_config = dict(
        family=None,
        termination=None,
        attr="NORM",
        use_sections=False,
        explain=True,
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
    family = family_factory(
        on_ents_only=on_ents_only,
        use_sections=use_sections,
    )

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

                assert getattr(ent._, modifier.key) == modifier.value, (
                    f"{modifier.key} labels don't match."
                )

                if not on_ents_only:
                    for token in ent:
                        assert getattr(token._, modifier.key) == modifier.value, (
                            f"{modifier.key} labels don't match."
                        )


def test_on_span(blank_nlp, family_factory):
    doc = blank_nlp("Le père du patient est asthmatique, le patient n'est pas malade.")
    doc.ents = [
        Span(doc, 5, 6, label="ent"),
        Span(doc, 12, 13, label="ent"),
    ]

    family = family_factory(on_ents_only=True)
    res = family.process(doc[1:13])
    assert [(ent.ent.text, bool(ent.family)) for ent in res.ents] == [
        ("asthmatique", True),
        ("malade", False),
    ]

    family = family_factory(on_ents_only=False)
    res = family.process(doc[1:13])
    assert [(t.token.text, t.family) for t in res.tokens] == [
        ("père", True),
        ("du", True),
        ("patient", True),
        ("est", True),
        ("asthmatique", True),
        (",", False),
        ("le", False),
        ("patient", False),
        ("n'", False),
        ("est", False),
        ("pas", False),
        ("malade", False),
    ]
