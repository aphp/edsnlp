import spacy
from pytest import fixture

from edsnlp.pipelines.core.endlines.functional import build_path
from edsnlp.pipelines.core.endlines.model import EndLinesModel

texts = [
    """Le patient est arrivé hier soir.
Il est accompagné par son fils

ANTECEDENTS
Il a fait une TS en 2010;
Fumeur, il est arreté il a 5 mois
Chirurgie de coeur en 2011
CONCLUSION
Il doit prendre
le medicament indiqué 3 fois par jour. Revoir médecin
dans 1 mois.
DIAGNOSTIC :

Antecedents Familiaux:
- 1. Père avec diabete

""",
    """J'aime le \nfromage...\n""",
]


@fixture
def model_path(tmp_path, blank_nlp):
    blank_nlp = spacy.blank("eds")

    # Train model
    docs = list(blank_nlp.pipe(texts))

    # Train and predict an EndLinesModel
    endlines = EndLinesModel(nlp=blank_nlp)
    df = endlines.fit_and_predict(docs)

    assert df["PREDICTED_END_LINE"].dtype == bool

    # path_model = os.path.join(tmp_path, "endlinesmodel.pkl")
    path_model = build_path(tmp_path, "endlinesmodel.pkl")
    endlines.save(path=path_model)

    return path_model


def test_endlines(blank_nlp, model_path):
    blank_nlp = spacy.blank("eds")

    # Use an existing trained model
    blank_nlp.add_pipe("endlines", config=dict(model_path=model_path))
    docs = list(blank_nlp.pipe(texts))

    assert [i for i, t in enumerate(docs[1]) if t.tag_ == "EXCLUDED"] == [3, 8]


def test_normalizer_endlines(blank_nlp, model_path):
    blank_nlp = spacy.blank("eds")

    # Use an existing trained model
    blank_nlp.add_pipe("normalizer")
    blank_nlp.add_pipe("endlines", config=dict(model_path=model_path))
    docs = list(blank_nlp.pipe(texts))

    assert [i for i, t in enumerate(docs[1]) if t.tag_ == "EXCLUDED"] == [3, 8]
