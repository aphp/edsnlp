from typing import Callable, List, Tuple

import spacy
from spacy.tokens import Doc
from thinc.api import Linear, Logistic, Model, chain
from thinc.types import Floats2d, Ints1d, Ragged, cast


@spacy.registry.architectures("edsnlp.qualifier_model.v1")
def create_qualifier_model(
    create_tensor: Model[List[Doc], Floats2d],
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain}):
        model = create_tensor >> classification_layer
    return model


@spacy.registry.architectures("edsnlp.classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None
) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return Linear(nO=nO, nI=nI) >> Logistic()


@spacy.registry.architectures("edsnlp.qualifier_tensor.v1")
def create_tensors(
    tok2vec: Model[List[Doc], List[Floats2d]],
    pooling: Model[Ragged, Floats2d],
) -> Model[List[Doc], Floats2d]:

    return Model(
        "instance_tensors",
        instance_forward,
        layers=[tok2vec, pooling],
        refs={"tok2vec": tok2vec, "pooling": pooling},
        init=instance_init,
    )


def instance_forward(
    model: Model[List[Doc], Floats2d], docs: List[Doc], is_train: bool
) -> Tuple[Floats2d, Callable]:
    pooling = model.get_ref("pooling")
    tok2vec = model.get_ref("tok2vec")

    all_instances = [list(doc.ents) for doc in docs]
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)

    ents = []
    lens = []

    for instances, tokvec in zip(all_instances, tokvecs):
        token_indices = []
        for ent in instances:
            token_indices.extend([i for i in range(ent.start, ent.end)])
            lens.append(ent.end - ent.start)
        ents.append(tokvec[token_indices])
    lengths = cast(Ints1d, model.ops.asarray(lens, dtype="int32"))
    entities = Ragged(model.ops.flatten(ents), lengths)
    pooled, bp_pooled = pooling(entities, is_train)

    def backprop(d_pooled: Floats2d) -> List[Doc]:
        d_ents = bp_pooled(d_pooled).data
        d_tokvecs = []
        ent_index = 0
        for doc_nr, instances in enumerate(all_instances):
            shape = tokvecs[doc_nr].shape
            d_tokvec = model.ops.alloc2f(*shape)
            count_occ = model.ops.alloc2f(*shape)
            for ent in instances:
                d_tokvec[ent.start : ent.end] += d_ents[ent_index]
                count_occ[ent.start : ent.end] += 1
                ent_index += ent.end - ent.start
            d_tokvec /= count_occ + 0.00000000001
            d_tokvecs.append(d_tokvec)

        d_docs = bp_tokvecs(d_tokvecs)
        return d_docs

    return pooled, backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model
