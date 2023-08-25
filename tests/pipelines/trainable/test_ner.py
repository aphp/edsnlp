import torch
from pytest import mark
from spacy.tokens import Span

import edsnlp
from edsnlp.core.component import TorchComponent


@mark.parametrize("ner_mode", ["independent", "joint", "marginal"])
def test_ner(ner_mode):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.ner",
        config=dict(
            embedding=nlp.get_pipe("eds.transformer"),
            mode=ner_mode,
            to_ents=True,
            to_span_groups="ner-preds",
        ),
    )
    doc = nlp(
        "L'aîné eut le Moulin, le second eut l'âne, et le plus jeune n'eut que le Chat."
    )
    # doc[0:2], doc[4:5], doc[6:8], doc[9:11], doc[13:16], doc[20:21]
    doc.ents = [
        Span(doc, 0, 2, "PERSON"),
        Span(doc, 4, 5, "GIFT"),
        Span(doc, 6, 8, "PERSON"),
        Span(doc, 9, 11, "GIFT"),
        Span(doc, 13, 16, "PERSON"),
        Span(doc, 20, 21, "GIFT"),
    ]

    ner: TorchComponent = nlp.get_pipe("eds.ner")  # type: ignore
    ner.update_labels(["PERSON", "GIFT"])
    batch = ner.make_batch([doc], supervision=True)
    batch = ner.collate(batch, device=torch.device("cpu"))
    batch = ner.module_forward(batch)

    assert batch["loss"] is not None
