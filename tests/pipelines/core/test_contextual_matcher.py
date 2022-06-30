from edsnlp.processing.helpers import rgetattr
from edsnlp.utils.examples import parse_example

example = """
Un <ent label_='Cancer' _.source='Cancer Solide' _.assigned={'metastase': 'metastase'}>Cancer</ent> métastasé de stade IV.
On a également un <ent label_='Cancer' _.source='Cancer Solide'>mélanome</ent> malin.
Aussi, un autre mélanome plutôt bénin.
Enfin, on remarque un <ent label_='Cancer' _.source='Lymphome' _.assigned={'stage': '3'}>lymphome de stade 3</ent>.
"""  # noqa: E501


def test_contextual(blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    terms = [
        "cancer",
        "tumeur",
    ]
    regex = [
        r"adeno(carcinom|[\s-]?k)",
        r"neoplas",
        r"melanom",
    ]
    benine = "benign|benin"
    stage = "stade (I{1,3}V?|[1234])"
    metastase = "(metasta)"
    cancer = dict(
        source="Cancer Solide",
        regex=regex,
        terms=terms,
        regex_attr="NORM",
        exclude=dict(
            regex=benine,
            window=3,
        ),
        assign=[
            dict(
                name="stage",
                regex=stage,
                window=(-10, 10),
                expand_entity=False,
            ),
            dict(
                name="metastase",
                regex=metastase,
                window=10,
                expand_entity=False,
            ),
        ],
    )
    lymphome = dict(
        source="Lymphome",
        regex=["lymphom", "lymphangio"],
        regex_attr="NORM",
    )
    patterns = [cancer, lymphome]

    blank_nlp.add_pipe(
        "eds.contextual-matcher",
        name="Cancer",
        config=dict(
            patterns=patterns,
        ),
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)

    assert len(doc.ents) == 3

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                rgetattr(ent, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."
