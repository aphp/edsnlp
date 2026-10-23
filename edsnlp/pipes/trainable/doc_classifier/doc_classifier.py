from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from spacy.tokens import Doc
from typing_extensions import NotRequired, TypedDict

from edsnlp.core.pipeline import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, TorchComponent
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.doc_classifier.heads import ClassificationHead
from edsnlp.pipes.trainable.embeddings.typing import (
    WordContextualizerComponent,
    WordEmbeddingComponent,
)

logger = logging.getLogger(__name__)

DocClassifierBatchInput = TypedDict(
    "DocClassifierBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[Dict[str, torch.Tensor]],
    },
)

DocClassifierBatchOutput = TypedDict(
    "DocClassifierBatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "logits": Optional[Dict[str, torch.Tensor]],
    },
)


class TrainableDocClassifier(
    TorchComponent[DocClassifierBatchOutput, DocClassifierBatchInput],
    BaseComponent,
):
    """
    The `eds.doc_classifier` component is a trainable document-level classifier.
    It predicts attributes of the document as a whole (its type, a principal
    diagnosis, the set of topics it covers) from a single pooled document
    embedding, and stores each prediction in a `Doc._` extension.

    The component is configured with one *head* per predicted attribute:
    [`eds.single_label_head`][edsnlp.pipes.trainable.doc_classifier.heads.SingleLabelHead]
    when a document has exactly one label,
    [`eds.multi_label_head`][edsnlp.pipes.trainable.doc_classifier.heads.MultiLabelHead]
    when it has a variable-length set of them.

    Architecture
    ------------
    A document embedding component (`eds.doc_pooler`) turns the document into a
    single vector, which is fed to each head. A head owns its own MLP, label set,
    loss and decoding logic, and writes to the `Doc._` attribute it is named
    after. At training time the per-head losses are aggregated into a single
    loss, weighted by each head's `loss_weight`.

    ![Document classifier architecture](/assets/images/doc-classifier.svg)

    Examples
    --------
    The pipeline below predicts the principal diagnosis of a hospital stay
    (exactly one per document, so a single-label head) and its associated
    diagnoses (a set, so a multi-label head). Each head is keyed by the name of
    the attribute it fills in.

    ```{ .python }
    import edsnlp, edsnlp.pipes as eds
    from edsnlp.pipes.trainable.doc_classifier.heads import (
        MultiLabelHead,
        SingleLabelHead,
    )

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=eds.doc_pooler(
                pooling_mode="mean",
                embedding=eds.transformer(
                    model="hf-internal-testing/tiny-random-bert",
                    window=128,
                    stride=96,
                ),
            ),
            heads={
                # the principal diagnosis: exactly one per document
                "dp": SingleLabelHead(labels=["C34", "I21"]),
                # the associated diagnoses: a set, of any size
                "das": MultiLabelHead(labels=["E11", "I10", "N18"], threshold=0.5),
            },
        ),
        name="doc_classifier",
    )

    doc = nlp("Compte rendu d'hospitalisation.")
    print(doc._.dp in ["C34", "I21"], isinstance(doc._.das, list))
    # Out: True True
    ```

    To predict a single attribute, keep only the corresponding head.

    `labels` may also be the path to a pickle file holding the list of labels,
    which is convenient for large label sets, or be left out entirely, in which
    case the labels are inferred from the gold data during `nlp.post_init(...)`.

    A document whose gold value is `None` for a head does not supervise it. A
    corpus annotated only for the principal diagnosis can therefore be mixed,
    in the same training run, with one annotated for everything.

    To train the model, refer to the [Training API](/training/training-api)
    documentation and to the
    [document classifier tutorial](/tutorials/training-doc-classifier).

    Extensions
    ----------
    The `eds.doc_classifier` pipeline declares one extension on the `Doc` object
    per head, named after the head:

    - `doc._.<head>`: the label predicted by the head, a string for a
    single-label head and a list of strings for a multi-label one.

    When `store_probs` is set, it also fills `doc._.prob`, a dict mapping each
    head name to a `{label: probability}` dict.

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline the component belongs to.
    name : str, default="doc_classifier"
        Component name.
    embedding : WordEmbeddingComponent or WordContextualizerComponent
        Document embedding component. Must expose an `output_size` attribute.
    heads : Dict[str, ClassificationHead]
        The heads of the classifier, keyed by the name of the `Doc._` attribute
        each one fills in. Every head also accepts the usual
        [head options][edsnlp.pipes.trainable.doc_classifier.heads.ClassificationHead]:
        a label set, an optional hidden block, class weights and a loss weight.
    store_probs : bool, default=False
        Whether to store the per-label probabilities of each head in
        `doc._.prob[<head>]`.
    probs_topk : Optional[int], default=None
        Only store the probabilities of the `probs_topk` most probable labels of
        each head. Useful when a head has thousands of labels.
    probs_threshold : Optional[float], default=None
        Only store the probabilities greater than or equal to `probs_threshold`.

    Authors and citation
    --------------------
    The `eds.doc_classifier` pipeline was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "doc_classifier",
        *,
        embedding: Union[WordEmbeddingComponent, WordContextualizerComponent],
        heads: Dict[str, ClassificationHead],
        store_probs: bool = False,
        probs_topk: Optional[int] = None,
        probs_threshold: Optional[float] = None,
    ):
        if not heads:
            raise ValueError("`heads` must be a non-empty mapping of head name -> head")

        # Plain (non-Module) attributes must be set before `super().__init__`,
        # which calls `set_extensions()`. Module attributes (embedding, heads)
        # can only be assigned after `nn.Module.__init__` has run.
        self.head_names: List[str] = list(heads)
        self.store_probs = store_probs
        self.probs_topk = probs_topk
        self.probs_threshold = probs_threshold

        super().__init__(nlp, name)
        self.embedding = embedding

        # `output_size` is part of the WordEmbeddingComponent contract, which
        # confit checks before we get here.
        self.embedding_size = self.embedding.output_size

        self.heads = nn.ModuleDict(heads)

        # Build heads whose labels are already known (e.g. loaded from a pickle).
        # Heads without labels yet are built later in `post_init` after scanning
        # the gold data.
        for head in self.heads.values():
            if head.num_classes is not None and not head.built:
                head.build(self.embedding_size)

    def set_extensions(self) -> None:
        """
        Register the `Doc._` extensions used to store predictions: one per
        head, named after it, and `prob` for the per-label probabilities.
        """
        super().set_extensions()
        for head_name in self.head_names:
            if not Doc.has_extension(head_name):
                Doc.set_extension(head_name, default=None)
        if not Doc.has_extension("prob"):
            Doc.set_extension("prob", default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]) -> None:
        """
        Finalize the heads once gold data is available.

        For every head whose label set is still unknown (i.e. no label list was
        provided at construction), the labels are inferred by scanning
        `gold_data`. Each head is then built (hidden block, classifier and
        loss) against the embedding output size.

        Parameters
        ----------
        gold_data : Iterable[Doc]
            Training documents carrying gold labels in `doc._.<head_name>`.
        exclude : Set[str]
            Set of already-initialized component ids, used to avoid
            re-initializing shared sub-components.
        """
        gold_data = list(gold_data) if not isinstance(gold_data, list) else gold_data
        for head_name, head in self.heads.items():
            if head.num_classes is None:
                head.set_labels_from_gold(gold_data, head_name)
                if head.num_classes is not None:
                    logger.info(
                        "Head %r initialized with %d classes.",
                        head_name,
                        head.num_classes,
                    )
            if not head.built:
                head.build(self.embedding_size)
        super().post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        """Preprocess a document for inference (embedding inputs only)."""
        return {"embedding": self.embedding.preprocess(doc)}

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess a document for training.

        Adds one target tensor per head under the key `targets_<head_name>`.
        A head whose gold attribute is missing (`None`) gets an ignored
        target, so a document only supervises the heads it actually has labels
        for (e.g. data labelled with the principal diagnosis only).
        """
        return {
            **self.preprocess(doc),
            **{
                f"targets_{head_name}": head.build_target(doc, head_name)
                for head_name, head in self.heads.items()
            },
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> DocClassifierBatchInput:
        """
        Collate a batch of preprocessed documents.

        Stacks the embedding inputs and, when present, the per-head target
        tensors.
        """
        batch_input: DocClassifierBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"])
        }
        collated_targets = {}
        for head_name in self.head_names:
            key = f"targets_{head_name}"
            if key in batch:
                collated_targets[head_name] = torch.stack(batch[key])
        if collated_targets:
            batch_input["targets"] = collated_targets
        return batch_input

    def forward(self, batch: DocClassifierBatchInput) -> DocClassifierBatchOutput:
        """
        Run the shared embedding and every head.

        In training mode (targets present) it returns the aggregated loss, a
        `loss_weight`-weighted mean over the heads supervised by at least one
        document of the batch. In inference mode it returns the raw per-head
        logits, which `postprocess` decodes.

        Parameters
        ----------
        batch : DocClassifierBatchInput
            Batched embedding inputs and, optionally, per-head targets.

        Returns
        -------
        DocClassifierBatchOutput
            `{"loss": tensor, "logits": None}` in training mode, or
            `{"loss": None, "logits": {head: tensor}}` in inference mode.
        """
        shared = self.embedding(batch["embedding"])["embeddings"]

        head_logits = {
            head_name: head(shared) for head_name, head in self.heads.items()
        }

        if "targets" in batch:
            total_loss = None
            total_weight = 0.0
            for head_name, head in self.heads.items():
                target = batch["targets"][head_name]
                supervised = head.is_supervised(target)
                if not supervised.any():
                    continue
                head_loss = head.loss_weight * head.compute_loss(
                    head_logits[head_name][supervised], target[supervised]
                )
                total_loss = head_loss if total_loss is None else total_loss + head_loss
                total_weight += head.loss_weight
            if total_loss is not None and total_weight > 0:
                total_loss = total_loss / total_weight
            return {"loss": total_loss, "logits": None}

        return {"loss": None, "logits": head_logits}

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: DocClassifierBatchOutput,
        input: DocClassifierBatchInput,
    ) -> Sequence[Doc]:
        """
        Decode the per-head logits and store predictions on the documents.

        Each head writes its prediction to `doc._.<head_name>`. When
        `store_probs` is set, the probabilities of the labels selected by
        `probs_topk` and `probs_threshold` are also written to
        `doc._.prob[<head_name>]`.

        Parameters
        ----------
        docs : Sequence[Doc]
            The documents processed in this batch.
        results : DocClassifierBatchOutput
            The forward output holding the per-head `logits`.
        input : DocClassifierBatchInput
            The collated batch (unused, kept for interface compatibility).

        Returns
        -------
        Sequence[Doc]
            The same documents with predictions stored in their extensions.
        """
        logits = results.get("logits")
        if logits is None:
            return docs

        for head_name, head in self.heads.items():
            head_logits = logits[head_name].detach().cpu()
            for doc, value in zip(docs, head.decode(head_logits)):
                setattr(doc._, head_name, value)
            if self.store_probs:
                for doc, probs in zip(docs, self._select_probs(head, head_logits)):
                    if doc._.prob is None:
                        doc._.prob = {}
                    doc._.prob[head_name] = probs

        return docs

    def _select_probs(
        self, head: ClassificationHead, logits: torch.Tensor
    ) -> List[Dict[Any, float]]:
        """
        Turn a batch of logits into one `{label: probability}` mapping per
        document, sorted by decreasing probability and restricted to the
        `probs_topk` most probable labels whose probability is at least
        `probs_threshold`.
        """
        probs = head.probs(logits)
        k = probs.size(-1)
        if self.probs_topk is not None:
            k = min(self.probs_topk, k)
        values, indices = probs.topk(k, dim=-1)
        results = []
        for row_values, row_indices in zip(values.tolist(), indices.tolist()):
            results.append(
                {
                    head.id2label.get(i, i): p
                    for p, i in zip(row_values, row_indices)
                    if self.probs_threshold is None or p >= self.probs_threshold
                }
            )
        return results

    def to_disk(self, path: Path, *, exclude: Set[str] = set()) -> Optional[Dict]:
        """
        Save the component to disk.

        Writes the labels of each head to `multi_head_data.pkl`; the head
        tensors themselves are saved by the parent `TorchComponent` along with
        the rest of the module tree. The other head parameters come from the
        config.

        Parameters
        ----------
        path : Path
            Directory to write to.
        exclude : Set[str]
            Set of already-saved object ids, used to avoid saving shared
            sub-components twice.
        """
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        os.makedirs(path, exist_ok=True)
        meta = {"heads": {name: head.state_meta() for name, head in self.heads.items()}}
        with open(path / "multi_head_data.pkl", "wb") as f:
            pickle.dump(meta, f)
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path: Path, exclude: Tuple = tuple()) -> None:
        """
        Load the component from disk.

        Restores the labels of each head from `multi_head_data.pkl` and
        rebuilds the head, so that the parameter shapes match before the parent
        `TorchComponent` loads the tensors.

        Parameters
        ----------
        path : Path
            Directory to read from.
        exclude : Tuple
            Already-loaded object ids, to avoid loading shared sub-components
            twice.
        """
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        with open(path / "multi_head_data.pkl", "rb") as f:
            meta = pickle.load(f)
        for head_name, head in self.heads.items():
            head.load_meta(meta["heads"][head_name])
            head.build(self.embedding_size)
        super().from_disk(path, exclude=exclude)
