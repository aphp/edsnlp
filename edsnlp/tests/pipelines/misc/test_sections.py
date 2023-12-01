from pytest import fixture

from edsnlp.pipelines.misc.sections import Sections, patterns
from edsnlp.utils.examples import parse_example

sections_text = (
    "Le patient est admis pour des douleurs dans le bras droit, "
    "mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNb"
    "WbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "<ent section=motif>Douleurs</ent> dans le bras droit.\n"
    "Pas d'anomalie détectée."
)

empty_sections_text = """
Antécédents :
Conclusion :
<ent section=conclusion>Patient</ent> va mieux

Au total:
sortie du patient
"""


def test_section_detection(doc):
    assert doc.spans["sections"]


@fixture
def sections_factory(blank_nlp):

    default_config = dict(
        sections=patterns.sections,
        add_patterns=True,
        attr="NORM",
        ignore_excluded=True,
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

    blank_nlp.add_pipe("normalizer")

    sections = sections_factory()

    text, entities = parse_example(example=sections_text)

    doc = blank_nlp(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, "generic") for ent in entities
    ]

    doc = sections(doc)

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                getattr(ent._, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."


def test_empty_sections(blank_nlp, sections_factory):

    blank_nlp.add_pipe("normalizer")

    sections = sections_factory()

    text, entities = parse_example(example=empty_sections_text)

    doc = blank_nlp(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
    ]

    doc = sections(doc)

    for section in doc.spans["sections"]:
        for ent in section.ents:
            ent._.section = section.label_

    for entity, ent in zip(entities, doc.ents):
        for modifier in entity.modifiers:
            assert (
                getattr(ent._, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."
