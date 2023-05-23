import spacy
from pytest import fixture, mark
from spacy.tokens import Span
from spacy.training import Example

from edsnlp.pipelines.trainable.nested_ner.factory import NESTED_NER_DEFAULTS
from edsnlp.utils.training import make_spacy_corpus_config, train


@fixture
def gold(blank_nlp):
    doc = blank_nlp.make_doc("Arret du ttt si folfox inefficace")

    # drug = "folfox"
    doc.ents = [Span(doc, 4, 5, "drug")]

    # event = "Arret"
    doc.spans["event"] = [Span(doc, 0, 1, "event")]

    # criteria = "si folfox inefficace"
    doc.spans["criteria"] = [Span(doc, 3, 6, "criteria")]

    return doc


@mark.parametrize("crf_mode", ["marginal", "independent", "joint"])
@mark.parametrize("lang", ["eds"], indirect=True)
def test_nested_ner_training(blank_nlp, gold, tmp_path, crf_mode):
    tmp_path.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("eds")
    nlp.add_pipe(
        "nested_ner",
        config={
            **NESTED_NER_DEFAULTS,
            "model": {
                **NESTED_NER_DEFAULTS["model"],
                "mode": crf_mode,
            },
        },
    )

    train(
        nlp,
        output_path=tmp_path,
        config=dict(
            **make_spacy_corpus_config(train_data=[gold], dev_data=[gold]),
            **{
                "training.max_steps": 10,
                "training.eval_frequency": 5,
            },
        ),
    )

    pred = nlp(gold.text)
    scores = nlp.pipeline[-1][1].score([Example(pred, gold)])

    assert scores["ents_f"] == 1.0
    assert len(pred.ents) == 1
    assert len(pred.spans["event"]) == 1
    assert len(pred.spans["criteria"]) == 1

    spacy.load(tmp_path / "model-last")
