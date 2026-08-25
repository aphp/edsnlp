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
from edsnlp.pipes.trainable.doc_classifier.heads import (  # noqa: F401
    ClassificationHead,
    FocalLoss,
)
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
    It predicts an attribute of the document as a whole — its type, a principal
    diagnosis, the set of topics it covers — from a single pooled document
    embedding, and stores the prediction in a `Doc._` extension.

    Most of the time you want to predict **one** attribute, and the component is
    configured with a single *head*: `eds.single_label_head` when a document has
    exactly one label, `eds.multi_label_head` when it has a variable-length set
    of them. Heads can then be combined to predict several attributes at once,
    over a shared embedding — see [Multiple heads](#multiple-heads) below.

    Architecture
    ------------
    A document embedding component (`eds.doc_pooler`) turns the document into a
    single vector, which is fed to each head. A head owns its own MLP, label set,
    loss and decoding logic, and writes to the `Doc._` attribute it is named
    after. At training time the per-head losses are aggregated into a single
    loss, weighted by each head's `loss_weight`.

    Examples
    --------
    Let us define a pipeline that predicts the type of a document, of which
    there is exactly one per document. This is the single-label case, so we use
    `eds.single_label_head`, keyed by the name of the attribute it fills in —
    here `doc._.doc_type`.

    ```{ .python }
    import edsnlp, edsnlp.pipes as eds
    from edsnlp.pipes.trainable.doc_classifier.heads import SingleLabelHead

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
                "doc_type": SingleLabelHead(
                    labels=["discharge_summary", "consultation", "lab_report"],
                    loss="ce",
                ),
            },
        ),
        name="doc_classifier",
    )

    doc = nlp("Compte rendu d'hospitalisation.")
    print(doc._.doc_type in ["discharge_summary", "consultation", "lab_report"])
    # Out: True
    ```

    When a document can carry *several* labels at once — say the topics it
    covers — use `eds.multi_label_head` instead. It is trained with a binary
    cross-entropy and predicts a **list** of labels, keeping those whose
    probability exceeds `threshold`.

    ```{ .python }
    import edsnlp, edsnlp.pipes as eds
    from edsnlp.pipes.trainable.doc_classifier.heads import MultiLabelHead

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
                "topics": MultiLabelHead(
                    labels=["diabetes", "hypertension", "kidney_failure"],
                    loss="bce",
                    selection="threshold",
                    threshold=0.5,
                ),
            },
        ),
        name="doc_classifier",
    )

    doc = nlp("Compte rendu d'hospitalisation.")
    print(isinstance(doc._.topics, list))
    # Out: True
    ```

    In both cases, `labels` may also be the path to a pickle file holding the
    list of labels — convenient for large label sets — or be left out entirely,
    in which case the labels are inferred from the gold data during
    `nlp.post_init(...)`.

    To train the model, refer to the [Training API](/training/training-api)
    documentation.

    Multiple heads
    --------------
    Adding a second entry to `heads` predicts a second attribute from the *same*
    document embedding, which is computed once and shared. This is cheaper than
    running two pipelines, and lets the tasks regularize each other.

    Two mechanisms are worth knowing about:

    - **Partial supervision.** A document whose gold value is `None` for a head
      simply does not supervise it. A corpus annotated only for the principal
      diagnosis can therefore be mixed, in the same training run, with one
      annotated for everything.
    - **Count heads.** Rather than a fixed `threshold`, a multi-label head can
      keep its top-`k` labels, `k` being predicted by a companion single-label
      head whose labels are the integers `0..K`. Point the multi-label head at
      it with `count_head`, and set `selection="topk"`.

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
                pooling_mode="attention",
                embedding=eds.transformer(
                    model="hf-internal-testing/tiny-random-bert",
                    window=128,
                    stride=96,
                ),
            ),
            heads={
                # the principal diagnosis: exactly one per document
                "dp": SingleLabelHead(labels=["C34", "I21"], loss="ce"),
                # the associated diagnoses: as many as `das_count` predicts
                "das": MultiLabelHead(
                    labels=["E11", "I10", "N18"],
                    loss="bce",
                    selection="topk",
                    count_head="das_count",
                    loss_weight=1.0,
                ),
                # how many associated diagnoses to keep
                "das_count": SingleLabelHead(
                    labels=[0, 1, 2, 3],
                    loss="ce",
                    loss_weight=0.5,
                ),
            },
        ),
        name="doc_classifier",
    )

    doc = nlp("Compte rendu d'hospitalisation.")
    print(len(doc._.das) == int(doc._.das_count))
    # Out: True
    ```

    Extensions
    ----------
    The `eds.doc_classifier` pipeline declares one extension on the `Doc` object
    per head, named after the head:

    - `doc._.<head>`: the label predicted by the head, a string for a
    single-label head and a list of strings for a multi-label one.

    Multi-label heads declare a second extension:

    - `doc._.<head>_alt`: the labels the head would have predicted with the
    *other* selection strategy (`topk` if `selection="threshold"`, and
    conversely). It lets both strategies be scored from a single inference pass,
    without retraining.

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline the component belongs to.
    name : str, default="doc_classifier"
        Component name.
    embedding : WordEmbeddingComponent or WordContextualizerComponent
        Document embedding component. Must expose an ``output_size`` attribute.
    heads : Dict[str, ClassificationHead]
        The heads of the classifier, keyed by the name of the `Doc._` attribute
        each one fills in. A single entry is the common case; see
        [Multiple heads](#multiple-heads).

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
    ):
        if not heads:
            raise ValueError("`heads` must be a non-empty mapping of head name -> head")

        # Plain (non-Module) attributes must be set before `super().__init__`,
        # which calls `set_extensions()`. Module attributes (embedding, heads)
        # can only be assigned after `nn.Module.__init__` has run.
        self.head_names: List[str] = list(heads)
        self._multilabel_names: List[str] = [
            name for name, head in heads.items() if getattr(head, "multilabel", False)
        ]
        for name in self._multilabel_names:
            count_head = heads[name].count_head
            if count_head is not None and count_head not in heads:
                raise ValueError(
                    f"Head {name!r} refers to a count head {count_head!r} that is "
                    f"not one of the heads ({', '.join(map(repr, heads))})."
                )

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
        Register the ``Doc._`` extensions used to store predictions.

        One extension per head (named after the head) is created, plus, for each
        multi-label head, a secondary ``<head>_alt`` extension holding the
        prediction obtained with the alternative selection strategy (used to
        compare ``threshold`` vs ``topk`` decoding offline).
        """
        super().set_extensions()
        for head_name in self.head_names:
            if not Doc.has_extension(head_name):
                Doc.set_extension(head_name, default=None)
        # Secondary extension to compare the alternative DAS selection strategy.
        for head_name in getattr(self, "_multilabel_names", []):
            alt = f"{head_name}_alt"
            if not Doc.has_extension(alt):
                Doc.set_extension(alt, default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]) -> None:
        """
        Finalize the heads once gold data is available.

        For every head whose label set is still unknown (i.e. no label list was
        provided at construction), the labels are inferred by scanning
        ``gold_data``. Each head is then built (hidden block, classifier and
        loss) against the embedding output size.

        Parameters
        ----------
        gold_data : Iterable[Doc]
            Training documents carrying gold labels in ``doc._.<head_name>``.
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

        Adds one target tensor per head under the key ``targets_<head_name>``.
        Heads whose gold attribute is missing (``None``) are skipped, so a
        document only supervises the heads it actually has labels for (e.g.
        synthetic data labelled with the principal diagnosis only).
        """
        preps = self.preprocess(doc)
        targets = {}
        for head_name, head in self.heads.items():
            target = head.build_target(doc, head_name)
            if target is not None:
                targets[f"targets_{head_name}"] = target
        return {**preps, **targets}

    def collate(self, batch: Dict[str, Sequence[Any]]) -> DocClassifierBatchInput:
        """
        Collate a batch of preprocessed documents.

        Stacks the embedding inputs and, when present, the per-head target
        tensors. A head's targets are only collated if every document in the
        batch produced one (true per data stream, since a stream supervises a
        fixed set of heads).
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
        ``loss_weight``-weighted mean over the heads that have targets. In
        inference mode it returns the raw per-head logits, which
        :meth:`postprocess` decodes (raw logits are kept so multi-label heads
        can use a companion count head for top-k selection).

        Parameters
        ----------
        batch : DocClassifierBatchInput
            Batched embedding inputs and, optionally, per-head targets.

        Returns
        -------
        DocClassifierBatchOutput
            ``{"loss": tensor, "logits": None}`` in training mode, or
            ``{"loss": None, "logits": {head: tensor}}`` in inference mode.
        """
        shared = self.embedding(batch["embedding"])["embeddings"]

        head_logits = {
            head_name: head(shared) for head_name, head in self.heads.items()
        }

        if "targets" in batch:
            total_loss = None
            total_weight = 0.0
            for head_name, head in self.heads.items():
                if head_name not in batch["targets"]:
                    continue
                logits = head_logits[head_name]
                target = batch["targets"][head_name]
                head_loss = head.loss_weight * head.compute_loss(logits, target)
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

        Single-label heads are decoded independently (``argmax``). Multi-label
        heads are decoded afterwards because they may consume a companion count
        head's prediction as the number of labels ``k`` for top-k selection. The
        chosen prediction is written to ``doc._.<head_name>``; for multi-label
        heads the alternative selection strategy is also written to
        ``doc._.<head_name>_alt`` for offline comparison.

        Parameters
        ----------
        docs : Sequence[Doc]
            The documents processed in this batch.
        results : DocClassifierBatchOutput
            The forward output holding the per-head ``logits``.
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

        logits = {name: t.detach().cpu() for name, t in logits.items()}

        # Single-label heads (incl. count heads) decode independently.
        decoded: Dict[str, List] = {}
        counts: Dict[str, List[int]] = {}
        for head_name, head in self.heads.items():
            if not getattr(head, "multilabel", False):
                decoded[head_name] = head.decode(logits[head_name])
                counts[head_name] = [
                    int(v) if isinstance(v, (int, float)) or str(v).isdigit() else 0
                    for v in decoded[head_name]
                ]

        # Multi-label heads may depend on a companion count head.
        for head_name, head in self.heads.items():
            if getattr(head, "multilabel", False):
                k = counts.get(head.count_head) if head.count_head else None
                decoded[head_name] = head.decode(logits[head_name], counts=k)
                # Store the alternative strategy for offline comparison.
                alt = head.decode_alt(logits[head_name], counts=k)
                for doc, value in zip(docs, alt):
                    setattr(doc._, f"{head_name}_alt", value)

        for head_name, values in decoded.items():
            for doc, value in zip(docs, values):
                setattr(doc._, head_name, value)

        return docs

    def to_disk(self, path: Path, *, exclude: Set[str] = set()) -> Optional[Dict]:
        """
        Save the component to disk.

        Writes the per-head non-tensor state (label mappings and configuration)
        to ``multi_head_data.pkl``; the head tensors themselves are saved by the
        parent :class:`TorchComponent` along with the rest of the module tree.

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
        meta = {
            "head_names": self.head_names,
            "heads": {name: head.state_meta() for name, head in self.heads.items()},
        }
        with open(path / "multi_head_data.pkl", "wb") as f:
            pickle.dump(meta, f)
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path: Path, exclude: Tuple = tuple()) -> None:
        """
        Load the component from disk.

        Restores the per-head label mappings and configuration from
        ``multi_head_data.pkl`` and rebuilds any head that was not already built
        at construction, so the parameter shapes match before the parent
        :class:`TorchComponent` loads the tensors.

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
        self.head_names = meta.get("head_names", self.head_names)
        for head_name, head_meta in meta.get("heads", {}).items():
            if head_name in self.heads:
                head = self.heads[head_name]
                head.load_meta(head_meta)
                if not head.built and head.num_classes is not None:
                    head.build(self.embedding_size)
        super().from_disk(path, exclude=exclude)
