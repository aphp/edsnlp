import sys
import os
from pytest import mark
import pytest
from spacy.tokens import Doc, Span

# Assurez-vous que le chemin vers edsnlp est en premier dans sys.path
sys.path.insert(0, "/home/pidoux/edsnlp")

# Importation des modules après avoir ajouté le chemin
import edsnlp
import edsnlp.pipes.core as eds

@mark.parametrize("use_sentences", [True, False])
@mark.parametrize("clean_rel", [True, False])
@mark.parametrize("proximity_method", ["sym", "right", "left", "middle", "start", "end"])
@mark.parametrize("max_dist", [1, 40, 100])
def test_relations(use_sentences, clean_rel, proximity_method, max_dist):
    dossier = "../../resources/relations/"  
    doc_iterator = edsnlp.data.read_standoff(dossier)
    corpus = list(doc_iterator)
    assert len(corpus) > 0
    for doc in corpus:
        assert isinstance(doc, Doc)
        for label in doc.spans:
            for span in doc.spans[label]:
                assert isinstance(span, Span)
                assert span.has_extension('rel')
                for rel in span._.rel:
                    assert isinstance(rel['target'], Span)
                    assert isinstance(rel['type'], str)
                    assert rel['type'] == 'Depend' or rel['type'] == 'inv_Depend'

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.relations",
        config={
            "scheme": os.path.join(dossier, "relations.json"),
            "use_sentences": use_sentences,  
            "clean_rel": clean_rel,  
            "proximity_method": proximity_method,  
            "max_dist": max_dist,  
        },
    )

    doc = nlp(corpus[0])

    for label in doc.spans:
        for span in doc.spans[label]:
            print(span, span._.rel)
            assert isinstance(span, Span)
            assert span.has_extension('rel')
            for rel in span._.rel:
                assert isinstance(rel['target'], Span)
                assert isinstance(rel['type'], str)
                assert rel['type'] == 'Depend' or rel['type'] == 'inv_Depend'
                
if __name__ == "__main__":
    pytest.main()
