from dummy_embeddings import DummyEmbeddings

import edsnlp.pipes as eds
from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.metrics.relations import RelationsMetric


def test_relation_candidates_and_metric():
    """
    Check label restrictions and duplicate pairs in relation candidates and scores
    """
    converter = MarkupToDocConverter()
    docs = [
        converter(prefix + "[Aspirin](drug) [pain](problem) [fever](other)")
        for prefix in ("", "Ignore ")
    ]
    for label_filter, expected in [
        (None, 2),
        ({"drug": {"problem"}}, 1),
        ({"drug": set()}, 0),
        ({"other": set()}, 2),
    ]:
        getter = {
            "head": {"ents": "drug"},
            "tail": {"ents": ["problem", "other"]},
            "labels": ["treats"],
            "label_filter": label_filter,
            "symmetric": False,
        }
        relation = eds.relation_detector_ffn(
            span_embedding=eds.span_pooler(embedding=DummyEmbeddings(dim=2)),
            candidate_getter=[getter, getter],
        )
        for doc in docs:
            doc.ents[0]._.rel = {"treats": set(doc.ents[1:])}
        prep = relation.preprocess_supervised(docs[0])
        assert prep["stats"]["relation_candidates"] == expected
        assert prep["rel_labels"] == [[True]] * expected
        assert prep["$getter"] == [0] * expected
        scores = RelationsMetric(
            candidate_getter=[getter, getter],
            filter_expr="not doc.text.startswith('Ignore')",
        )(docs, docs)["micro"]
        assert scores["tp"] == scores["positives"] == scores["support"] == expected
