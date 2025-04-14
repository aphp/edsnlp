import os

import pytest

import edsnlp
import edsnlp.pipes as eds
from edsnlp.utils.examples import parse_example
from edsnlp.utils.extensions import rgetattr

os.environ["CONFIT_DEBUG"] = "1"

EXAMPLES = [
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>cancer métastasé au stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': ['metastase']}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer métastasé au stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer métastasé au stade 3 voire au stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>stade 3</ent> voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>stade 3</ent> voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au stade 3 voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au stade 3 voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un cancer métastasé au stade 3 voire au <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>stade 4</ent>.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': ['metastase']}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': ['3', '4'], 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': ['metastase']}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '3', 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': ['metastase']}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
    """
    Le patient présente une métastasis sur un <ent label_='Cancer' _.source='Solide' _.assigned={'stage': '4', 'metastase': 'metastase'}>cancer </ent>métastasé au stade 3 voire au stade 4.
    """,  # noqa: E501
]

ALL_PARAMS = [
    (True, True, None, None),
    (True, True, None, "keep_first"),
    (True, True, None, "keep_last"),
    (True, True, "keep_first", None),
    (True, True, "keep_first", "keep_first"),
    (True, True, "keep_first", "keep_last"),
    (True, True, "keep_last", None),
    (True, True, "keep_last", "keep_first"),
    (True, True, "keep_last", "keep_last"),
    (True, False, None, None),
    (True, False, None, "keep_first"),
    (True, False, None, "keep_last"),
    (True, False, "keep_first", None),
    (True, False, "keep_first", "keep_first"),
    (True, False, "keep_first", "keep_last"),
    (True, False, "keep_last", None),
    (True, False, "keep_last", "keep_first"),
    (True, False, "keep_last", "keep_last"),
    (False, True, None, None),
    (False, True, None, "keep_first"),
    (False, True, None, "keep_last"),
    (False, True, "keep_first", None),
    (False, True, "keep_first", "keep_first"),
    (False, True, "keep_first", "keep_last"),
    (False, True, "keep_last", None),
    (False, True, "keep_last", "keep_first"),
    (False, True, "keep_last", "keep_last"),
    (False, False, None, None),
    (False, False, None, "keep_first"),
    (False, False, None, "keep_last"),
    (False, False, "keep_first", None),
    (False, False, "keep_first", "keep_first"),
    (False, False, "keep_first", "keep_last"),
    (False, False, "keep_last", None),
    (False, False, "keep_last", "keep_first"),
    (False, False, "keep_last", "keep_last"),
]


@pytest.mark.parametrize("params,example", list(zip(ALL_PARAMS, EXAMPLES)))
def test_contextual(blank_nlp, params, example):
    include_assigned, replace_entity, reduce_mode_stage, reduce_mode_metastase = params

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
        source="Solide",
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
                replace_entity=replace_entity,
                reduce_mode=reduce_mode_stage,
            ),
            dict(
                name="metastase",
                regex=metastase,
                window=10,
                reduce_mode=reduce_mode_metastase,
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
            label="Cancer",
            patterns=patterns,
            include_assigned=include_assigned,
        ),
    )

    text, entities = parse_example(example=example)

    doc = blank_nlp(text)

    assert len(doc.ents) == len(entities)

    for entity, ent in zip(entities, doc.ents):
        for modifier in entity.modifiers:
            assert rgetattr(ent, modifier.key) == modifier.value, (
                f"{modifier.key} labels don't match."
            )


def test_contextual_matcher_include(blank_nlp):
    if not isinstance(blank_nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")
    blank_nlp.add_pipe(
        eds.quantities(
            span_setter=["sizes"],
            quantities=["size"],
        ),
    )
    blank_nlp.add_pipe(
        eds.contextual_matcher(
            name="tumor_size",
            label="tumor_size",
            assign_as_span=True,
            patterns=[
                dict(
                    source="tumor_size",
                    terms=["cancer", "tumeur"],
                    regex_attr="NORM",
                    include=dict(regex="mamm", window="sents[-1:1]"),
                    assign=dict(
                        name="size",
                        span_getter="sizes",
                        reduce_mode="first",
                        required=True,
                    ),
                )
            ],
        ),
    )
    doc = blank_nlp("""\
Bilan mammaire:
La tumeur est de 3 cm.
Tumeur au pied sans changement.
Tumeur mammaire benigne.
""")
    assert len(doc.ents) == 1
    ent = doc.ents[0]
    assert ent.label_ == "tumor_size"
    assert ent._.assigned["size"]._.value.cm == 3


# Checks https://github.com/aphp/edsnlp/issues/394
def test_contextual_matcher_exclude_outside():
    import edsnlp
    import edsnlp.pipes as eds

    asa_pattern = r"\basa\b ?:? ?([1-5]|[A-Z]{1,3})"
    exclude_asa_ttt = r"5"
    asa = dict(
        source="asa",
        regex=asa_pattern,
        regex_attr="NORM",
        exclude=dict(regex=exclude_asa_ttt, window=-5),
    )

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.contextual_matcher(patterns=[asa], label="asa"))

    doc = nlp("ASA 5")
    assert str(doc.ents) == "(ASA 5,)"

    doc = nlp("5 ASA 5")
    assert str(doc.ents) == "()"
