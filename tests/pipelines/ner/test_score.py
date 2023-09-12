import re

from edsnlp.pipelines.ner.scores import Score
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
SOFA maximum : <ent score_name=sofa score_value=12 score_method=Maximum>12</ent>.


CR-URG.
PRIORITE: <ent score_name=emergency_priority score_value=2>2</ent>: 2 - Urgence relative.
GEMSA: (<ent score_name=emergency_gemsa score_value=2>2</ent>) Patient non convoque sortant apres consultation
CCMU: Etat clinique jugé stable avec actes diag ou thérapeutiques ( <ent score_name=emergency_ccmu score_value=2>2</ent> )


CONCLUSION

La patiente est atteinte d'un carcinome mammaire infiltrant de type non spécifique, de grade 2 de malignité selon Elston et Ellis
<ent score_name=elston_ellis score_value=2>(architecture : 3 + noyaux : 3 + mitoses : 1)</ent>.

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
        name="TestScore",
        score_name="TestScore",
        regex=[r"test+score"],
        attr="NORM",
        ignore_excluded=True,
        ignore_space_tokens=False,
        value_extract=r"(\d+)",
        score_normalization=testscore_normalization,
        window=4,
        flags=re.S,
    )

    text, entities = parse_example(example=example)

    blank_nlp.add_pipe("eds.charlson")
    blank_nlp.add_pipe("eds.sofa")
    blank_nlp.add_pipe("eds.elston_ellis")
    blank_nlp.add_pipe("eds.emergency_priority")
    blank_nlp.add_pipe("eds.emergency_ccmu")
    blank_nlp.add_pipe("eds.emergency_gemsa")

    doc = blank_nlp(text)
    doc = testscore(doc)

    for entity, ent in zip(entities, doc.ents):

        for modifier in entity.modifiers:
            assert (
                getattr(ent._, modifier.key) == modifier.value
            ), f"{modifier.key} labels don't match."
