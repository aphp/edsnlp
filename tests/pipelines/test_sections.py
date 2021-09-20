from typing import List
from edsnlp.utils.examples import parse_example

from edsnlp.pipelines.sections import terms, Sections

from pytest import fixture, mark

sections_text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "<ent section=motif>Douleurs</ent> dans le bras droit.\n"
    "Pas d'anomalie détectée."
)


def test_section_detection(doc):
    assert doc.spans["sections"]


@fixture
def sections_factory(blank_nlp):

    default_config = dict(
        sections=terms.sections,
        add_patterns=True,
        attr="NORM",
        fuzzy=False,
        fuzzy_kwargs=None,
    )

    def factory(**kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return Sections(
            nlp=blank_nlp,
            **config,
        )

    return factory


def test_sections(blank_nlp, sections_factory):

    sections = sections_factory()

    text, entities = parse_example(example=sections_text)

    doc = blank_nlp(text)
    doc.ents = [doc.char_span(ent.start_char, ent.end_char) for ent in entities]

    doc = sections(doc)

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
