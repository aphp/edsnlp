from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import foldedtensor as ft
import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

import edsnlp.data
from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.base import BaseSpanAttributeClassifierComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
)
from edsnlp.pipes.trainable.layers.metric import Metric
from edsnlp.utils.collections import (
    batch_compress_dict,
    decompress_dict,
    ld_to_dl,
)
from edsnlp.utils.span_getters import (
    SpanGetter,
    SpanGetterArg,
    get_spans,
    get_spans_with_group,
)

SpanLinkerBatchInput = TypedDict(
    "SpanLinkerBatchInput",
    {
        "embedding": BatchInput,
        "span_labels": torch.Tensor,
        "concepts": NotRequired[torch.Tensor],
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
mask: torch.BoolTensor
    Mask of the sequences
spans: torch.LongTensor
    2d tensor of n_spans * (doc_idx, ner_label_idx, begin, end)
targets: NotRequired[List[torch.LongTensor]]
    list of 2d tensor of n_spans * n_combinations (1 hot)
"""


class TrainableSpanLinker(
    TorchComponent[BatchOutput, SpanLinkerBatchInput],
    BaseSpanAttributeClassifierComponent,
):
    """
    The `eds.span_linker` component is a trainable span concept predictor, typically
    used to match spans in the text with concepts in a knowledge base. This task is
    known as "Entity Linking", "Named Entity Disambiguation" or "Normalization" (the
    latter is mostly used in the biomedical machine learning community).

    !!! warning "Entity Linking vs Named Entity Recognition"

        Entity Linking is the task of linking existing entities to their concept in a
        knowledge base, while Named Entity Recognition is the task of detecting spans in
        the text that correspond to entities. The `eds.span_linker` component should
        therefore be used after the Named Entity Recognition step (e.g. using the
        `eds.ner_crf` component).

    How it works
    ------------

    To perform this task, this components compare the embedding of a given query span
    (e.g. "aspirin") with the embeddings in the knowledge base, where
    each embedding represents a concept (e.g. "B01AC06"), and selects the most similar
    embedding and returns its concept id. This comparison is done using either:

    - the cosine similarity between the input and output embeddings (recommended)
    - a simple dot product

    We filter out the concepts that are not relevant for a given query by using groups.
    For each span to link, we use its label to select a group of concepts to compare
    with. For example, if the span is labeled as "drug", we only compare it with
    concepts that are drugs. These concepts groups are inferred from the training data
    when running the `post_init` method, or can be provided manually using the
    `pipe.update_concepts(concepts, mapping, [embeddings])` method. If a label is not
    found in the mapping, the span is compared with all concepts.

    We support comparing entity queries against two kind of references : either the
    embeddings of the concepts themselves (`reference_mode = "concept"`), or the
    embeddings of the synonyms of the concepts (`reference_mode = "synonym"`).

    ### Synonym similarity

    When performing span linking in `synonym` mode, the span linker embedding matrix
    contains one embedding vector per concept per synonym, and each embedding maps to
    the concept of its synonym. This mode is slower and more memory intensive, since you
    have to store multiple embeddings per concept, but it can yield good results
    in zero-shot scenarios (see example below).

    <figure markdown>
      ![Synonym similarity](/assets/images/synonym_span_linker.png)
      <figcaption>Entity linking based on synonym similarity</figcaption>
    </figure>

    ### Concept similarity

    In `concept` mode, the span linker embedding matrix contains one embedding vector
    per concept : imagine a single vector that approximately averages all the synonyms
    of a concept (e.g. B01AC06 = average of "aspirin", "acetyl-salicylic acid", etc.).
    This mode is faster and more memory efficient, but usually requires that the
    concept weights are fine-tuned.

    <figure markdown>
      ![Concept similarity](/assets/images/class_span_linker.png)
      <figcaption>Entity linking based on concept similarity</figcaption>
    </figure>

    Examples
    --------

    Here is how you can use the `eds.span_linker` component to link spans without
    training, in `synonym` mode. You will still need to pre-compute the embeddings of
    the target synonyms.


    First, initialize the component:

    ```python
    import pandas as pd
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_linker(
            rescale=20.0,
            threshold=0.8,
            metric="cosine",
            reference_mode="synonym",
            probability_mode="sigmoid",
            span_getter=["ents"],
            embedding=eds.span_pooler(
                hidden_size=128,
                embedding=eds.transformer(
                    model="prajjwal1/bert-tiny",
                    window=128,
                    stride=96,
                ),
            ),
        ),
        name="linker",
    )
    ```

    We will assume you have a list of synonyms with their concept and label with the
    columns:

    - `STR`: synonym text
    - `CUI`: concept id
    - `GRP`: label.

    All we need to do is to initialize the component with the synonyms and that's it !
    Since we have set `init_weights` to True, and we are in `synonym` mode, the
    embeddings of the synonyms will be stored in the component and used to compute the
    similarity scores

    ```{ .python .no-check }
    synonyms_df = pd.read_csv("synonyms.csv")

    def make_doc(row):
        doc = nlp.make_doc(row["STR"])
        span = doc[:]
        span.label_ = row["GRP"]
        doc.ents = [span]
        span._.cui = row["CUI"]
        return doc

    nlp.post_init(
        edsnlp.data.from_pandas(
            synonyms_df,
            converter=make_doc,
        )
    )
    ```

    Now, you can now use it in a text:
    ```{ .python .no-check }
    doc = nlp.make_doc("Aspirin is a drug")
    span = doc[0:1]  # "Aspirin"
    span.label_ = "Drug"
    doc.ents = [span]

    doc = nlp(doc)
    print(doc.ents[0]._.cui)
    # "B01AC06"
    ```

    To use the `eds.span_linker` component in `class` mode, we refer to the following
    repository: [deep_multilingual_normalization](
    https://github.com/percevalw/deep_multilingual_normalization) based on the work
    of [@wajsburt2021medical].

    Parameters
    ----------
    nlp: PipelineProtocol
        Spacy vocabulary
    name: str
        Name of the component
    embedding : SpanEmbeddingComponent
        The word embedding component
    metric : Literal["cosine", "dot"] = "cosine"
        Whether to compute the cosine similarity between the input and output embeddings
        or the dot product.
    rescale : float
        Rescale the output cosine similarities by a constant factor.
    threshold : float
        Threshold probability to consider a concept as valid
    attribute : str
        The attribute to store the concept id
    reference_mode : Literal["concept", "synonym"]
        Whether to compare the embeddings with the concepts embeddings (one per concept)
        or the synonyms embeddings (one per concept per synonym). See above for more
        details.
    span_getter : SpanGetterArg
        How to extract the candidate spans to predict or train on.
    context_getter : Optional[SpanGetterArg]
        What context to use when computing the span embeddings (defaults to the entity
        only, so no context)
    probability_mode : Literal["softmax", "sigmoid"]
        Whether to compute the probabilities using a softmax or a sigmoid function.
        This will also determine the loss function to use, either cross-entropy or
        binary cross-entropy.

        !!! warning "Subsetting the concepts"

            The probabilities returned in `softmax` mode depend on the number of
            concepts (as an extreme cas, if you have only one concept, its softmax
            probability will always be 1). This is why we recommend using the `sigmoid`
            mode in which the probabilities are computed independently for each concept.
    init_weights : bool
        Whether to initialize the weights of the component with the embeddings of
        the entities of the docs provided to the `post_init` method.
        How this is done depends on the `reference_mode` parameter:

        - `concept`: the embeddings are averaged
        - `synonym`: the embeddings are stored as is

        By default, this is set to `True` if `reference_mode` is `synonym`, and
        `False` otherwise.

    Authors and citation
    --------------------
    The `eds.span_linker` component was developed by AP-HP's Data Science team.

    The deep learning concept-based architecture was adapted from
    [@wajsburt2021medical].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "span_linker",
        *,
        embedding: SpanEmbeddingComponent,
        metric: Literal["cosine", "dot"] = "cosine",
        rescale: float = 20,
        threshold: float = 0.5,
        attribute: str = "cui",
        span_getter: SpanGetterArg = None,
        context_getter: Optional[SpanGetterArg] = None,
        reference_mode: Literal["concept", "synonym"] = "concept",
        probability_mode: Literal["softmax", "sigmoid"] = "sigmoid",
        init_weights: bool = True,
    ):
        self.attribute = attribute

        sub_span_getter = getattr(embedding, "span_getter", None)
        if sub_span_getter is not None and span_getter is None:  # pragma: no cover
            span_getter = sub_span_getter
        span_getter = span_getter or {"ents": True}
        sub_context_getter = getattr(embedding, "context_getter", None)
        if (
            sub_context_getter is not None and context_getter is None
        ):  # pragma: no cover
            context_getter = sub_context_getter

        super().__init__(
            nlp,
            name,
            span_getter=span_getter,
        )

        self.embedding = embedding
        self.concepts: List = []
        self.concepts_to_idx: Dict[str, int] = {}
        self.span_labels_to_idx: Dict[str, int] = {}
        self.threshold = threshold
        self.reference_mode = reference_mode
        self.probability_mode = probability_mode
        self.init_weights = init_weights
        self.context_getter: SpanGetter = context_getter or self.span_getter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.classifier = Metric(
                in_features=self.embedding.output_size,
                out_features=0,
                num_groups=0,
                rescale=rescale,
                metric=metric,
            )

    @property
    def attributes(self) -> List[str]:
        return [self.attribute]

    def to_disk(self, path, *, exclude=set()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        # We save the bindings as a pickle file since values can be arbitrary objects
        os.makedirs(path, exist_ok=True)
        data_path = path / "vocab.json"
        with open(data_path, "w") as file:
            json.dump(
                {
                    "concepts": self.concepts,
                    "span_labels": list(self.span_labels_to_idx.keys()),
                },
                file,
            )
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path, exclude=tuple()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        data_path = path / "vocab.json"
        with open(data_path, "r") as f:
            data = json.load(f)
        self.span_labels_to_idx = {lab: i for i, lab in enumerate(data["span_labels"])}
        self.classifier.groups.data = torch.zeros(
            len(self.span_labels_to_idx),
            len(data["concepts"]),
            dtype=torch.bool,
        )
        self.update_concepts(concepts=data["concepts"])
        self.set_extensions()
        super().from_disk(path, exclude=exclude)

    def set_extensions(self):
        if not Span.has_extension(self.attribute):
            Span.set_extension(self.attribute, default=None)
        if not Span.has_extension("prob"):
            # Builtin mutable defaults will be copied the first time they're accessed
            Span.set_extension("prob", default={})
        super().set_extensions()

    def update_concepts(
        self,
        *,
        concepts: Optional[Sequence[str]] = None,
        mapping: Optional[Dict[str, Iterable[str]]] = None,
        embeds: Optional[torch.Tensor] = None,
    ):
        old_concepts_to_idx = self.concepts_to_idx
        device = self.device
        if mapping is not None:
            mapping = {k: set(v) for k, v in mapping.items()}
            concepts = (
                sorted(set(cui for cuis in mapping.values() for cui in cuis))
                if concepts is None
                else concepts
            )

        assert concepts is not None, "You must provide either `concepts` or `mapping`"

        self.concepts_to_idx = {
            cui: idx for idx, cui in enumerate(sorted(set(concepts)))
        }

        new_lin = torch.nn.Linear(
            self.classifier.in_features,
            len(concepts)
            if self.reference_mode == "synonym"
            else len(self.concepts_to_idx),
        ).to(device)
        if embeds is not None:
            if self.reference_mode == "synonym":
                embeds = embeds.to(new_lin.weight)
                new_lin.weight.data = embeds
            else:
                old_weight_norm = (
                    self.classifier.weight.norm(dim=-1).mean().item()
                    if self.classifier.weight.shape[0]
                    else new_lin.weight.norm(dim=-1).mean().item()
                )
                cui_indices = torch.as_tensor(
                    [self.concepts_to_idx[c] for c in concepts],
                    dtype=torch.long,
                ).to(device)
                new_lin.weight.data.zero_().index_add_(
                    0, cui_indices, embeds.to(new_lin.weight)
                )
                del embeds
                new_lin.weight.data = (
                    F.normalize(
                        new_lin.weight.data
                        / cui_indices.bincount(minlength=len(self.concepts_to_idx))
                        .to(device)
                        .unsqueeze(-1)
                    )
                    * old_weight_norm
                )
        else:
            common = sorted(set(self.concepts_to_idx) & set(old_concepts_to_idx))
            old_index = [old_concepts_to_idx[label] for label in common]
            new_index = [self.concepts_to_idx[label] for label in common]
            new_lin.weight.data[new_index] = self.classifier.weight.data[old_index]

        self.concepts = (
            concepts if self.reference_mode == "synonym" else list(self.concepts_to_idx)
        )
        self.classifier.weight.data = new_lin.weight.data
        self.classifier.out_features = self.classifier.weight.shape[0]

        if mapping is not None:
            mapping = {k: set(v) for k, v in mapping.items()}
            self.classifier.groups.data = torch.tensor(
                [[True] * len(self.concepts)]
                + [
                    [concept in label_concepts for concept in self.concepts]
                    for span_label, label_concepts in mapping.items()
                ],
                device=device,
            )
            span_labels = [None] + list(mapping.keys())
            span_labels_to_idx = {label: idx for idx, label in enumerate(span_labels)}
            self.span_labels_to_idx = span_labels_to_idx

    def compute_init_features(
        self,
        docs: Iterable[Doc],
        with_embeddings=True,
    ) -> Tuple[List[str], List[str], Optional[torch.Tensor]]:
        embedding = self.embedding

        def prepare_batch(docs, device=None):
            res = {}
            if with_embeddings:
                device = device or embedding.device
                spans = [list(get_spans(doc, self.span_getter)) for doc in docs]
                batch = [
                    embedding.preprocess(
                        doc,
                        spans=doc_spans,
                        contexts=list(get_spans(doc, self.context_getter)),
                    )
                    for doc, doc_spans in zip(docs, spans)
                ]
                spans = [span for doc in spans for span in doc]
                batch = decompress_dict(list(batch_compress_dict(batch)))
                inputs = embedding.collate(batch)
                inputs = embedding.batch_to_device(inputs, device=device)
                res["inputs"] = inputs
            else:
                spans = [
                    span for doc in docs for span in get_spans(doc, self.span_getter)
                ]
            batch_concepts = [s._.get(self.attribute) for s in spans]
            batch_labels = [s.label_ for s in spans]
            res.update(
                {
                    "concepts": batch_concepts,
                    "labels": batch_labels,
                }
            )
            return res

        def run_forward(batch):
            res = {
                "concepts": batch["concepts"],
                "labels": batch["labels"],
            }
            if with_embeddings:
                res["embeds"] = F.normalize(
                    embedding(batch["inputs"])["embeddings"]
                ).cpu()
            return res

        # Loop over the data to compute the embeddings and collect the concepts
        docs = edsnlp.data.from_iterable(docs)
        results = ld_to_dl(
            docs.map_gpu(
                forward=run_forward,
                prepare_batch=prepare_batch,
            )
            if with_embeddings
            else docs.map_batches(prepare_batch)
        )
        all_embeds = (
            torch.cat(results["embeds"], dim=0).to(self.device)
            if "embeds" in results
            else None
        )
        all_concepts = [c for batch in results["concepts"] for c in batch]
        all_labels = [c for batch in results["labels"] for c in batch]
        if not len(all_concepts):
            warnings.warn("Did not find any concept when scanning the gold data.")

        return all_concepts, all_labels, all_embeds

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)

        concepts, labels, embeds = self.compute_init_features(
            gold_data, with_embeddings=self.init_weights
        )
        labels_to_cuis = defaultdict(set)
        for label, cui in zip(labels, concepts):
            if cui is not None:
                labels_to_cuis[label].add(cui)

        self.update_concepts(
            concepts=concepts,
            mapping=labels_to_cuis,
            embeds=embeds,
        )

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        spans_and_groups = list(get_spans_with_group(doc, self.span_getter))
        spans = [span for span, _ in spans_and_groups]
        groups = [group for _, group in spans_and_groups]
        contexts = list(get_spans(doc, self.context_getter))
        embedding = self.embedding.preprocess(doc, spans=spans, contexts=contexts)

        # "$" prefix means this field won't be accessible from the outside.
        return {
            "embedding": embedding,
            "span_labels": [self.span_labels_to_idx.get(s.label_, 0) for s in spans],
            "$spans": spans,
            "$groups": groups,
        }

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        preps = self.preprocess(doc)
        spans = preps["$spans"]
        return {
            **preps,
            "concepts": [self.concepts_to_idx.get(span._.cui, -100) for span in spans],
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanLinkerBatchInput:
        span_labels = ft.as_folded_tensor(
            batch["span_labels"],
            dtype=torch.long,
            full_names=("sample", "span"),
            data_dims=("span",),
        ).as_tensor()
        collated: SpanLinkerBatchInput = {
            "span_labels": span_labels,
            "embedding": self.embedding.collate(batch["embedding"]),
        }
        if "concepts" in batch:
            collated["concepts"] = ft.as_folded_tensor(
                batch["concepts"],
                dtype=torch.long,
                full_names=("sample", "span"),
                data_dims=("span",),
            ).as_tensor()
        return collated

    def forward(self, inputs: SpanLinkerBatchInput) -> BatchOutput:
        span_embeds = self.embedding(inputs["embedding"])["embeddings"].refold("span")

        span_labels = inputs["span_labels"]
        targets = inputs.get("concepts")

        scores = self.classifier(
            inputs=span_embeds,
            group_indices=span_labels,
        )
        concepts = loss = top_probs = None
        if targets is not None:
            #  TRAINING
            num_classes = self.classifier.weight.shape[0]
            loss = (
                F.binary_cross_entropy_with_logits(
                    scores,
                    F.one_hot(targets, num_classes=num_classes).float(),
                    reduction="sum",
                )
                / num_classes
                if self.probability_mode == "sigmoid"
                else F.cross_entropy(scores, targets, reduction="sum")
            )
        else:
            # PREDICTION
            probs = (
                torch.sigmoid(scores)
                if self.probability_mode == "sigmoid"
                else torch.softmax(scores, -1)
            )
            top_probs, concepts = probs.max(-1)

        return {
            "loss": loss,
            "concepts": concepts,
            "probs": top_probs,
        }

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: BatchOutput,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[Doc]:
        ents = {doc: {} for doc in docs}
        spans = [span for sample in inputs for span in sample["$spans"]]
        groups = [group for sample in inputs for group in sample["$groups"]]

        for span, group, concept_idx, prob in zip(
            spans,
            groups,
            results["concepts"].tolist(),
            results["probs"].tolist(),
        ):
            span._.set(
                self.attribute,
                self.concepts[concept_idx] if prob >= self.threshold else None,
            )
            span._.prob[self.attribute] = {self.concepts[concept_idx]: prob}
            if group == "ents":
                ents[span.doc][span.start] = span
        for doc, doc_ents in ents.items():
            if doc_ents:
                doc.ents = {
                    **{span.start: span for span in doc.ents},
                    **doc_ents,
                }.values()
        return docs
