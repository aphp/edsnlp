import pytest
from pytest import mark
from spacy.tokens import Span

import edsnlp


@mark.parametrize(
    "ner_mode,window",
    [
        ("independent", 1),
        ("joint", 0),
        ("joint", 5),
        ("marginal", 0),
    ],
)
def test_ner(ner_mode, window):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config=dict(
            embedding=nlp.get_pipe("transformer"),
            mode=ner_mode,
            target_span_getter=["ents", "ner-preds"],
            window=window,
        ),
    )
    nlp.pipes.ner.compute_confidence_score = True
    ner = nlp.get_pipe("ner")
    ner.update_labels([])
    doc = nlp(
        "L'aîné eut le Moulin, le second eut l'âne, et le plus jeune n'eut que le Chat."
    )
    ner.labels = ["LOC", "ORG"]
    # doc[0:2], doc[4:5], doc[6:8], doc[9:11], doc[13:16], doc[20:21]
    doc.ents = [
        Span(doc, 0, 2, "PERSON"),
        Span(doc, 4, 5, "GIFT"),
        Span(doc, 6, 8, "PERSON"),
        Span(doc, 9, 11, "GIFT"),
        Span(doc, 13, 16, "PERSON"),
        Span(doc, 20, 21, "GIFT"),
    ]
    with pytest.warns() as record:
        nlp.post_init([doc])
    assert len(record) == 1
    assert record[0].message.args[0] == (
        "The labels inferred from the data are different from the labels passed to "
        "the component. Differing labels are ['GIFT', 'LOC', 'ORG', 'PERSON']"
    )

    ner = nlp.get_pipe("ner")
    ner.update_labels(["PERSON", "GIFT"])
    batch = ner.prepare_batch([doc], supervision=True)
    batch = ner(batch)

    list(ner.pipe([doc]))

    assert batch["loss"] is not None
