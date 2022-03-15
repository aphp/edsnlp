from edsnlp.pipelines.core.advanced import AdvancedRegex
from edsnlp.utils.examples import parse_example

example = """
Faible fracture du pied.
Le patient présente plusieurs <ent label_=fracture>fêlures</ent>.
Présence de fractures de fatigues.
"""


def test_advanced(blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    regex_config = dict(
        fracture=dict(
            regex=[r"fracture", r"felure"],
            attr="NORM",
            before_exclude="petite|faible",
            after_exclude="legere|de fatigue",
        )
    )

    advanced_regex = AdvancedRegex(
        nlp=blank_nlp,
        regex_config=regex_config,
        window=5,
        verbose=True,
        ignore_excluded=False,
        attr="TEXT",
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)

    doc = advanced_regex(doc)

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                getattr(ent, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."


def test_attr_default(blank_nlp):
    text = "Le patient présente plusieurs <ent label_=fracture>fêlures</ent>."

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    regex_config = dict(
        fracture=dict(
            regex=[r"fracture", r"felures?"],
            before_exclude="petite|faible",
            after_exclude="legere|de fatigue",
        )
    )

    advanced_regex = AdvancedRegex(
        nlp=blank_nlp,
        regex_config=regex_config,
        window=5,
        verbose=True,
        ignore_excluded=False,
        attr="NORM",
    )

    text, entities = parse_example(example=text)

    doc = blank_nlp(text)

    doc = advanced_regex(doc)

    assert len(doc.ents) == 1
    assert doc.ents[0].text == "fêlures"
