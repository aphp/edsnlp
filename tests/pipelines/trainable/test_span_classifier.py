import spacy
from pytest import fixture, mark
from spacy.tokens import Span
from spacy.training import Corpus, Example

from edsnlp.pipelines.trainable.span_qualifier.factory import SPAN_QUALIFIER_DEFAULTS
from edsnlp.utils.training import make_spacy_corpus_config, train

if not Span.has_extension("label"):
    Span.set_extension("label", default=None)


if not Span.has_extension("event_type"):
    Span.set_extension("event_type", default=None)


if not Span.has_extension("test_negated"):
    Span.set_extension("test_negated", default=False)


@fixture
def gold(blank_nlp):
    doc1 = blank_nlp.make_doc("Arret du ttt si folfox inefficace")

    doc1.spans["sc"] = [
        # drug = "folfox"
        Span(doc1, 4, 5, "drug"),
        # event = "Arret"
        Span(doc1, 0, 1, "event"),
        # criteria = "si"
        Span(doc1, 3, 4, "criteria"),
    ]
    doc1.spans["sc"][0]._.test_negated = False
    doc1.spans["sc"][1]._.test_negated = True
    doc1.spans["sc"][2]._.test_negated = False
    doc1.spans["sc"][1]._.event_type = "stop"

    doc1.spans["sent"] = [Span(doc1, 0, 6, "sent")]

    doc2 = blank_nlp.make_doc("Début du traitement")

    span = Span(doc2, 0, 1, "event")
    doc2.ents = [
        # drug = "Début"
        span,
    ]
    span._.test_negated = False
    span._.event_type = "start"

    doc2.spans["sent"] = [Span(doc2, 0, 3, "sent")]

    return [doc1, doc2]


@spacy.registry.readers.register("test-span-classification-corpus")
class SpanClassificationCorpus(Corpus):
    def _make_example(self, nlp, reference, gold_preproc: bool):
        pred = reference.copy()
        pred.user_data = {
            key: value
            for key, value in pred.user_data.items()
            if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "._.")
        }
        return Example(
            pred,
            reference,
        )


@mark.parametrize("lang", ["eds"], indirect=True)
def test_span_qualifier_label_training(gold, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("eds")
    nlp.add_pipe(
        "span_qualifier",
        config={
            **SPAN_QUALIFIER_DEFAULTS,
            "qualifiers": ("label_",),
            "on_ents": False,
            "on_span_groups": True,
            "model": {
                **SPAN_QUALIFIER_DEFAULTS["model"],
            },
        },
    )

    train(
        nlp,
        output_path=tmp_path,
        config=dict(
            **make_spacy_corpus_config(
                train_data=gold,
                dev_data=gold,
                reader="test-span-classification-corpus",
            ),
            **{
                "training.max_steps": 10,
                "training.eval_frequency": 5,
                # "training.optimizer.learn_rate": 0,
            },
        ),
    )
    nlp = spacy.load(tmp_path / "model-best")

    pred = gold[0].copy()
    pred.spans["sc"] = [
        Span(span.doc, span.start, span.end, "ent") for span in pred.spans["sc"]
    ]
    pred.user_data = {
        key: value
        for key, value in pred.user_data.items()
        if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "._.")
    }
    pred = nlp(pred)
    scores = nlp.pipeline[-1][1].score([Example(pred, gold[0])])
    assert [span.label_ for span in pred.spans["sc"]] == ["drug", "event", "criteria"]
    assert scores["qual_f"] == 1.0


@mark.parametrize("lang", ["eds"], indirect=True)
def test_span_qualifier_constrained_training(gold, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("eds")
    nlp.add_pipe(
        "span_qualifier",
        config={
            **SPAN_QUALIFIER_DEFAULTS,
            "candidate_getter": {
                "@misc": "eds.candidate_span_qualifier_getter",
                "qualifiers": ("_.test_negated", "_.event_type"),
                "label_constraints": {"_.event_type": ("event",)},
                "on_ents": False,
                "on_span_groups": ("sc",),
            },
            "model": SPAN_QUALIFIER_DEFAULTS["model"],
        },
    )

    train(
        nlp,
        output_path=tmp_path,
        config=dict(
            **make_spacy_corpus_config(
                train_data=gold,
                dev_data=gold,
                reader="test-span-classification-corpus",
            ),
            **{
                "training.max_steps": 5,
                "training.eval_frequency": 5,
            },
        ),
    )
    nlp = spacy.load(tmp_path / "model-best")

    pred = gold[0].copy()
    pred.user_data = {
        key: value
        for key, value in pred.user_data.items()
        if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "._.")
    }
    assert [span._.test_negated for span in pred.spans["sc"]] == [False, False, False]
    pred = nlp(pred)
    scores = nlp.pipeline[-1][1].score([Example(pred, gold[0])])
    assert [s._.test_negated for s in pred.spans["sc"]] == [False, True, False]
    assert [s._.event_type for s in pred.spans["sc"]] == [None, "stop", None]
    assert scores["qual_f"] == 1.0
