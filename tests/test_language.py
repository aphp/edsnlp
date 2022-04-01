import pytest
import spacy
from spacy.lang.fr.lex_attrs import like_num


def test_eds_tokenizer_handles_long_text():
    text = """Témoin interne : + ; témoin externe : +
- Récepteurs aux œstrogènes : tous les élements sont marqués (3+).
- Récepteurs à la progestérone : 40% sont marqués (intensité 2+).
- Anti-Cerb B2: 0% des cellules carcinomateuses présentent
un marquage membranaire complet.
CONCLUSION`: ========
-Carcinome mammaire infiltrant du quadrant inféro-externe,
de 25 mm size de grade II de
malignité selon Elston et Ellis (3+2+1), sans composante
Score ACR5 de chaque coté`'.
On se donne rendez-vous pour le 23/11/1967.
"""
    nlp = spacy.blank("eds")
    tokens = nlp(text)
    assert (
        [t.text_with_ws for t in tokens]
        == """Témoin |interne |: |+ |; |témoin |externe |: |+|
|- |Récepteurs |aux |œstrogènes |: |tous |les |élements |sont |marqués |(|3|+|)|.|
|- |Récepteurs |à |la |progestérone |: |40|% |sont |marqués |(|intensité |2|+|)|.|
|- |Anti|-|Cerb |B|2|: |0|% |des |cellules |carcinomateuses |présentent|
|un |marquage |membranaire |complet|.|
|CONCLUSION|`|: |=|=|=|=|=|=|=|=|
|-|Carcinome |mammaire |infiltrant |du |quadrant |inféro|-|externe|,|
|de |25 |mm |size |de |grade |II |de|
|malignité |selon |Elston |et |Ellis |(|3|+|2|+|1|)|, |sans |composante|
|Score |ACR|5 |de |chaque |coté|`|'|.|
|On |se |donne |rendez|-|vous |pour |le |23|/|11|/|1967|.|
""".split(
            "|"
        )
    )


@pytest.mark.parametrize("word", ["onze", "onzième"])
def test_eds_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())
