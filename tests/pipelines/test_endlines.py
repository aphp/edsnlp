import os

from pytest import fixture

from edsnlp.pipelines.endlines.endlinesmodel import EndLinesModel
from edsnlp.pipelines.endlines.functional import build_path

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


def test_set_spans(blank_nlp):
    # Train model
    docs = list(blank_nlp.pipe(texts))

    # Train and predict an EndLinesModel
    endlines = EndLinesModel(nlp=blank_nlp)
    df = endlines.fit_and_predict(docs)

    # Test set_spans function
    endlines.set_spans(docs, df)
    doc_example = docs[1]
    assert "new_lines" in doc_example.spans.keys()


def test_endlines(blank_nlp, model_path):

    # Use an existing trained model
    blank_nlp.add_pipe("endlines", config=dict(model_path=model_path))
    docs = list(blank_nlp.pipe(texts))

    assert docs[0][0]._.end_line is None
    assert docs[1].spans["new_lines"][0].label_ == "space"


def test_normalizer_endlines(blank_nlp, model_path):

    # Use an existing trained model
    blank_nlp.add_pipe("normalizer", config=dict(endlines=dict(model_path=model_path)))
    docs = list(blank_nlp.pipe(texts))

    assert docs[0][0]._.end_line is None
    assert docs[1].spans["new_lines"][0].label_ == "space"
