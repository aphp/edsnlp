from edsnlp.pipelines.core.advanced import AdvancedRegex
from edsnlp.utils.examples import parse_example

example = """
Un Cancer métastasé de stade IV.
On a également un mélanome malin.
Aussi, un autre mélanome plutôt bénin
Enfin, on remarque un lymphome de stade 3.
"""


def test_advanced(blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    terms = [
        "cancer",
        "tumeur",
    ]
    regex = [
        "adeno(carcinom|[\s-]?k)",
        "neoplas",
        "melanom",
    ]
    benine = "benign|benin"
    stage = "stade (I{1,3}V?|[1234])"
    metastase = "(metasta)"
    cancer = dict(
        source="Cancer solide",
        regex=regex,
        terms=terms,
        regex_attr="NORM",
        exclude=dict(
            after=dict(
                regex=benine,
                window=3,
            )
        ),
        assign=dict(
            after=dict(
                regex=dict(
                    stage=stage,
                    metastase=metastase,
                )
            )
        ),
    )
    lymphome = dict(
        source="Lymphome",
        regex=["lymphom", "lymphangio"],
        regex_attr="NORM",
        exclude=dict(
            after=dict(
                regex=benine,
                window=3,
            )
        ),
        assign=dict(
            after=dict(
                regex=dict(
                    stage=stage,
                    metastase=metastase,
                )
            )
        ),
    )
    patterns = [cancer, lymphome]

    blank_nlp.add_pipe(
        "eds.contextual-matcher",
        config=dict(
            patterns=patterns,
            name="Cancer",
        ),
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)

    assert len(doc.ents) == 3
    return

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
