from typing import List

from pytest import fixture, mark

from edsnlp.pipelines.qualifiers.history import History
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    "Antécédents d'<ent history_=ATCD>AVC</ent>.",
    "atcd <ent history_=ATCD>chirurgicaux</ent> : aucun.",
    "Le patient est <ent history_=CURRENT>fumeur</ent>.",
    # Les sections ne sont pas utilisées par défaut
    (
        "\nv Antecedents :\n- <ent history_=CURRENT>appendicite</ent>\n"
        "v Motif :\n<ent history_=CURRENT>malaise</ent>"
    ),
]


@fixture
def history_factory(blank_nlp):

    default_config = dict(
        history=None,
        termination=None,
        use_sections=False,
        attr="LOWER",
        explain=True,
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return History(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("use_sections", [True, False])
@mark.parametrize("on_ents_only", [True, False])
def test_history(blank_nlp, history_factory, on_ents_only, use_sections):

    history = history_factory(on_ents_only, use_sections=use_sections)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = history(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert bool(ent._.history_cues) == (modifier.value in {"ATCD", True})

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
