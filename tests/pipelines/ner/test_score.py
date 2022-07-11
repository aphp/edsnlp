# noqa: F401
import re

from edsnlp.pipelines.ner.scores import Score

# from edsnlp.pipelines.ner.scores.charlson import patterns as charlson_terms
# from edsnlp.pipelines.ner.scores.sofa import patterns as sofa_terms
from edsnlp.utils.examples import parse_example

example = """
CR-REA.
Charlson à l'admission: <ent score_name=charlson score_value=8>8</ent>.
- Charlson:
- IMC: 21
Cette phrase teste un score qui s'appelle TestScore.
La seule valeur admissible est 0.
testScore de 1.
TestScore de <ent score_name=TestScore score_value=0>0</ent>.
Testons également un autre score.
SOFA maximum : <ent score_name=SOFA score_value=12 score_method=Maximum>12</ent>.


CR-URG.
PRIORITE: <ent score_name=emergency.priority score_value=2>2</ent>: 2 - Urgence relative.
GEMSA: (<ent score_name=emergency.gemsa score_value=2>2</ent>) Patient non convoque sortant apres consultation
CCMU: Etat clinique jugé stable avec actes diag ou thérapeutiques ( <ent score_name=emergency.ccmu score_value=2>2</ent> )

"""  # noqa: E501


def test_scores(blank_nlp):

    blank_nlp.add_pipe(
        "eds.normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    def testscore_normalization(raw_score: str):
        if raw_score is not None and int(raw_score) == 0:
            return int(raw_score)

    testscore = Score(
        blank_nlp,
        score_name="TestScore",
        regex=[r"test+score"],
        attr="NORM",
        ignore_excluded=True,
        after_extract=r"(\d+)",
        score_normalization=testscore_normalization,
        window=4,
        flags=re.S,
    )

    text, entities = parse_example(example=example)

    blank_nlp.add_pipe("charlson")
    blank_nlp.add_pipe("SOFA")
    blank_nlp.add_pipe("emergency.priority")
    blank_nlp.add_pipe("emergency.ccmu")
    blank_nlp.add_pipe("emergency.gemsa")

    doc = blank_nlp(text)
    doc = testscore(doc)

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:

            assert (
                getattr(ent._, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."
