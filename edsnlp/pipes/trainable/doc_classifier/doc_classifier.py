import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd
import torch
import torch.nn as nn
from spacy.tokens import Doc
from typing_extensions import Literal, NotRequired, TypedDict

import edsnlp
from edsnlp.core.pipeline import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, TorchComponent
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    WordContextualizerComponent,
    WordEmbeddingComponent,
)

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
        "labels": Optional[Dict[str, torch.Tensor]],
    },
)


@edsnlp.registry.misc.register("focal_loss")
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.

    Parameters
    ----------
    alpha : torch.Tensor or float, optional
        Class weights. If None, no weighting is applied
    gamma : float, default=2.0
        Focusing parameter. Higher values give more weight to hard examples
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TrainableDocClassifier(
    TorchComponent[DocClassifierBatchOutput, DocClassifierBatchInput],
    BaseComponent,
):
    """
    The `eds.doc_classifier` component is a trainable document-level classifier.
    In this context, document classification consists in predicting one or more
    categorical labels at the **document level** (e.g. diagnosis code, discharge
    status, or any metadata derived from the whole document).

    Unlike span classification, where predictions are attached to spans, the
    document classifier attaches predictions to the `Doc` object itself.

    Architecture
    ------------
    The model performs multi-head document classification by:

    1. Calling a word/document embedding component `eds.doc_pooler`
       to compute a pooled embedding for the document.
    2. Feeding the pooled embedding into one or more classification heads.
       Each head is defined by a linear layer (optionally preceded by a
       head-specific hidden layer with activation, dropout, and layer norm).
    3. Computing independent logits for each head.
    4. Training with a per-head loss (CrossEntropy or Focal), optionally using
       class weights to handle imbalance.
    5. Aggregating head losses into a single training loss (simple average).
    6. During inference, assigning the predicted label for each head to
       `doc._.labels[head_name]`.

    Each classification head is independent, so different tasks (e.g.
    predicting ICD-10 category vs. mortality flag) can be trained jointly
    on the same pooled embeddings.

    Examples
    --------
    To create a document classifier component:

    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            label_attr=["icd10", "mortality"],
            labels={
                "icd10": "data/path_to_label_list_icd10.pkl",
                "mortality": "data/path_to_label_list_mortality.pkl",
            },
            num_classes={
                "icd10": 1000,
                "mortality": 2,
            },
            class_weights={
                "icd10": "data/path_to_class_weights_icd10.pkl",
                "mortality": "data/path_to_class_weights_mortality.pkl",
            },
            embedding=eds.doc_pooler(
                pooling_mode="attention",
                embedding=eds.transformer(
                    model="almanach/camembertav2-base",
                    window=256,
                    stride=128,
                ),
            ),
            hidden_size=1024,
            activation_mode="relu",
            dropout_rate={
                "icd10": 0.05,
                "mortality": 0.2,
            },
            layer_norm=True,
            loss="ce",
        ),
        name="doc_classifier",
    )
    ```

    After training, predictions are stored in the `Doc` object:

    ```python
    doc = nlp("Patient was admitted with pneumonia and discharged alive.")
    print(doc._.icd10, doc._.mortality)
    # J18 alive
    ```

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The spaCy/edsnlp pipeline the component belongs to.
    name : str, default="doc_classifier"
        Component name.
    embedding : WordEmbeddingComponent or WordContextualizerComponent
        Embedding component (e.g. transformer + pooling).
        Must expose an `output_size` attribute.
    label_attr : List[str]
        List of head names. Each head corresponds to a document-level attribute
        (e.g. `["icd10", "mortality"]`).
    num_classes : dict of str -> int, optional
        Number of classes for each head. If not provided, inferred from labels.
    label2id : dict of str -> dict[str,int], optional
        Per-head mapping from label string to integer ID.
    id2label : dict of str -> dict[int,str], optional
        Reverse mapping (ID -> label string).
    loss : {"ce", "focal"} or dict[str, {"ce","focal"}], default="ce"
        Loss type, either shared or per-head.
    labels : dict of str -> str (path), optional
        Paths to pickle files containing label sets for each head.
    class_weights : dict of str -> str (path), optional
        Paths to pickle files containing class frequency dicts
        (converted into class weights).
    hidden_size : int or dict[str,int], optional
        Hidden layer size (before classifier), shared or per-head.
        If None, no hidden layer is used.
    activation_mode : {"relu","gelu","silu"} or dict[str,str], default="relu"
        Activation function for hidden layers, shared or per-head.
    dropout_rate : float or dict[str,float], default=0.0
        Dropout rate after activation, shared or per-head.
    layer_norm : bool or dict[str,bool], default=False
        Whether to apply layer normalization in hidden layers, shared or per-head.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "doc_classifier",
        *,
        embedding: Union[WordEmbeddingComponent, WordContextualizerComponent],
        label_attr: List[str],
        num_classes: Optional[Dict[str, int]] = None,
        label2id: Optional[Dict[str, Dict[str, int]]] = None,
        id2label: Optional[Dict[str, Dict[int, str]]] = None,
        loss: Union[Literal["ce", "focal"], Dict[str, Literal["ce", "focal"]]] = "ce",
        labels: Optional[Dict[str, str]] = None,
        class_weights: Optional[Dict[str, str]] = None,
        hidden_size: Optional[Union[int, Dict[str, int]]] = None,
        activation_mode: Union[
            Literal["relu", "gelu", "silu"], Dict[str, Literal["relu", "gelu", "silu"]]
        ] = "relu",
        dropout_rate: Optional[Union[float, Dict[str, float]]] = 0.0,
        layer_norm: Optional[Union[bool, Dict[str, bool]]] = False,
    ):
        if not isinstance(label_attr, list) or len(label_attr) == 0:
            raise ValueError("label_attr must be a non-empty list of strings")

        self.label_attr: List[str] = label_attr
        self.head_names = label_attr

        self.num_classes = num_classes or {}
        self.label2id = label2id or {head: {} for head in self.head_names}
        self.id2label = id2label or {head: {} for head in self.head_names}

        self.labels_from_pickle = {}
        if labels:
            for head_name, labels_path in labels.items():
                if head_name in self.head_names:
                    head_labels = pd.read_pickle(labels_path)
                    self.labels_from_pickle[head_name] = head_labels
                    self.num_classes[head_name] = len(head_labels)

        self.class_weights = {}
        if class_weights:
            for head_name, weights_path in class_weights.items():
                if head_name in self.head_names:
                    self.class_weights[head_name] = pd.read_pickle(weights_path)

        if isinstance(loss, str):
            self.loss_config = {head: loss for head in self.head_names}
        else:
            self.loss_config = loss

        if isinstance(hidden_size, (int, type(None))):
            self.hidden_size_config = {head: hidden_size for head in self.head_names}
        else:
            self.hidden_size_config = hidden_size

        if isinstance(activation_mode, str):
            self.activation_mode_config = {
                head: activation_mode for head in self.head_names
            }
        else:
            self.activation_mode_config = activation_mode

        if isinstance(dropout_rate, (float, type(None))):
            self.dropout_rate_config = {head: dropout_rate for head in self.head_names}
        else:
            self.dropout_rate_config = dropout_rate

        if isinstance(layer_norm, bool):
            self.layer_norm_config = {head: layer_norm for head in self.head_names}
        else:
            self.layer_norm_config = layer_norm

        super().__init__(nlp, name)
        self.embedding = embedding

        if not hasattr(self.embedding, "output_size"):
            raise ValueError(
                "The embedding component must have an 'output_size' attribute."
            )
        self.embedding_size = self.embedding.output_size

        if any(head in self.num_classes for head in self.head_names):
            self.build_classifiers()

    def build_classifiers(self):
        """
        Build classification heads for each task.

        For every head in `self.head_names`, creates:
        - An optional hidden layer (`Linear + activation + dropout [+ layer norm]`).
        - A final linear classifier projecting to `num_classes[head_name]`.

        All heads are stored in `nn.ModuleDict`s for modularity.
        """
        self.classifiers = nn.ModuleDict()
        self.hidden_layers = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()

        for head_name in self.head_names:
            if head_name in self.num_classes:
                hidden_size = self.hidden_size_config.get(head_name)

                if hidden_size:
                    self.hidden_layers[head_name] = torch.nn.Linear(
                        self.embedding_size, hidden_size
                    )

                    activation_mode = self.activation_mode_config.get(head_name, "relu")
                    self.activations[head_name] = {
                        "relu": nn.ReLU(),
                        "gelu": nn.GELU(),
                        "silu": nn.SiLU(),
                    }[activation_mode]

                    if self.layer_norm_config.get(head_name, False):
                        self.norms[head_name] = nn.LayerNorm(hidden_size)

                    dropout_rate = self.dropout_rate_config.get(head_name, 0.0)
                    self.dropouts[head_name] = nn.Dropout(dropout_rate)

                    classifier_input_size = hidden_size
                else:
                    classifier_input_size = self.embedding_size

                self.classifiers[head_name] = torch.nn.Linear(
                    classifier_input_size, self.num_classes[head_name]
                )

    def _compute_class_weights(
        self, freq_dict: Dict[str, int], label2id: Dict[str, int]
    ) -> torch.Tensor:
        """
        Compute class weights from a frequency dictionary.

        Parameters
        ----------
        freq_dict : dict[str, int]
            Mapping from label string to its frequency.
        label2id : dict[str, int]
            Mapping from label string to class index.

        Returns
        -------
        torch.Tensor
            A weight vector aligned with label indices, where each weight is
            proportional to the inverse of the label frequency.
        """
        total_samples = sum(freq_dict.values())
        weights = torch.zeros(len(label2id))

        for label, freq in freq_dict.items():
            if label in label2id:
                weight = total_samples / (len(label2id) * freq)
                weights[label2id[label]] = weight

        return weights

    def set_extensions(self) -> None:
        """
        Register custom spaCy extensions for storing predictions.

        For each head in `self.head_names`, adds an attribute
        `doc._.<head_name>` if it does not already exist.
        """
        super().set_extensions()
        for head_name in self.head_names:
            if not Doc.has_extension(head_name):
                Doc.set_extension(head_name, default={})

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        """
        Finalize initialization after gold data is available.

        - Builds label mappings (`label2id`, `id2label`) for each head if missing.
        - Infers label sets from pickle files or scans gold data.
        - Builds classifiers once `num_classes` are known.
        - Initializes loss functions per head (CrossEntropy or Focal).

        Parameters
        ----------
        gold_data : Iterable[Doc]
            Training documents containing gold labels.
        exclude : set
            Components to exclude from initialization.
        """
        for head_name in self.head_names:
            if not self.label2id[head_name]:
                if head_name in self.labels_from_pickle:
                    labels = self.labels_from_pickle[head_name]
                else:
                    labels = set()
                    for doc in gold_data:
                        label = getattr(doc._, head_name, None)
                        if isinstance(label, str):
                            labels.add(label)

                if labels:
                    self.label2id[head_name] = {
                        label: i for i, label in enumerate(labels)
                    }
                    self.id2label[head_name] = {
                        i: label for i, label in enumerate(labels)
                    }
                    self.num_classes[head_name] = len(labels)
                    print(f"Head '{head_name}': {self.num_classes[head_name]} classes")

        self.build_classifiers()

        self.loss_fns = {}
        for head_name in self.head_names:
            weight_tensor = None
            if head_name in self.class_weights:
                weight_tensor = self._compute_class_weights(
                    self.class_weights[head_name], self.label2id[head_name]
                )
                print(f"Head '{head_name}' - Using class weights: {weight_tensor}")

            loss_type = self.loss_config.get(head_name, "ce")
            if loss_type == "ce":
                self.loss_fns[head_name] = torch.nn.CrossEntropyLoss(
                    weight=weight_tensor
                )
            elif loss_type == "focal":
                self.loss_fns[head_name] = FocalLoss(
                    alpha=weight_tensor, gamma=2.0, reduction="mean"
                )
            else:
                raise ValueError(f"Unknown loss for head '{head_name}': {loss_type}")

        print("Loss functions initialized")
        super().post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess a single document for inference.

        Parameters
        ----------
        doc : Doc
            Input spaCy/edsnlp `Doc`.

        Returns
        -------
        dict
            Dictionary containing the pooled embedding of the document.
        """
        return {"embedding": self.embedding.preprocess(doc)}

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess a single document for training.

        Adds gold labels for each head to the embedding dict, mapping labels
        to integer indices when possible.

        Parameters
        ----------
        doc : Doc
            Input document with gold labels stored in `doc._.<head_name>`.

        Returns
        -------
        dict
            Dictionary with:
            - `"embedding"` : document embedding
            - `"targets_<head>"` : gold target tensor for each head
        """
        preps = self.preprocess(doc)
        targets = {}

        for head_name in self.head_names:
            label = getattr(doc._, head_name, None)
            if label is None:
                raise ValueError(
                    f"Document does not have a gold label in 'doc._.{head_name}'"
                )

            if isinstance(label, str) and head_name in self.label2id:
                if label not in self.label2id[head_name]:
                    raise ValueError(
                        f"Label '{label}' not in label2id for head '{head_name}'."
                    )
                label = self.label2id[head_name][label]

            targets[head_name] = torch.tensor(label, dtype=torch.long)
        return {
            **preps,
            **{
                f"targets_{head_name}": targets[head_name]
                for head_name in self.head_names
            },
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> DocClassifierBatchInput:
        """
        Collate a batch of preprocessed documents.

        Combines embeddings and per-head target tensors into a single batch.

        Parameters
        ----------
        batch : dict
            A list of per-document dicts returned by `preprocess_supervised`.

        Returns
        -------
        DocClassifierBatchInput
            Batched embeddings and optional targets.
        """
        embeddings = self.embedding.collate(batch["embedding"])
        batch_input: DocClassifierBatchInput = {"embedding": embeddings}

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
        Forward pass through the model.

        - Computes shared embeddings.
        - Applies each classification head independently.
        - Computes per-head losses (if targets provided) and averages them.
        - Otherwise, returns predicted class indices for each head.

        Parameters
        ----------
        batch : DocClassifierBatchInput
            Batched embeddings and optional targets.

        Returns
        -------
        DocClassifierBatchOutput
            Dict with `"loss"` (training mode) or `"labels"` (inference mode).
        """
        pooled = self.embedding(batch["embedding"])
        shared_embeddings = pooled["embeddings"]

        head_logits = {}
        for head_name in self.head_names:
            if head_name in self.classifiers:
                x = shared_embeddings

                if head_name in self.hidden_layers:
                    x = self.hidden_layers[head_name](x)
                    x = self.activations[head_name](x)
                    if head_name in self.norms:
                        x = self.norms[head_name](x)
                    x = self.dropouts[head_name](x)

                head_logits[head_name] = self.classifiers[head_name](x)

        output: DocClassifierBatchOutput = {}

        if "targets" in batch:
            head_losses = []
            for head_name in self.head_names:
                if head_name in head_logits and head_name in batch["targets"]:
                    logits = head_logits[head_name]
                    targets = batch["targets"][head_name].to(logits.device)

                    loss_fn = self.loss_fns[head_name]
                    if hasattr(loss_fn, "weight") and loss_fn.weight is not None:
                        loss_fn.weight = loss_fn.weight.to(logits.device)

                    head_loss = loss_fn(logits, targets)
                    head_losses.append(head_loss)

            output["loss"] = torch.stack(head_losses).mean() if head_losses else None
            output["labels"] = None
        else:
            head_predictions = {
                head_name: torch.argmax(logits, dim=-1)
                for head_name, logits in head_logits.items()
            }
            output["loss"] = None
            output["labels"] = head_predictions

        return output

    def postprocess(self, docs, results, input):
        """
        Attach predictions to documents after inference.

        For each head, predicted labels are mapped back to strings using
        `id2label` and stored in `doc._.<head_name>`.

        Parameters
        ----------
        docs : list[Doc]
            Documents processed by the pipeline.
        results : dict
            Output of the forward pass (`"labels"`).
        input : dict
            Input batch (unused).

        Returns
        -------
        list[Doc]
            The same documents with predictions stored in extensions.
        """
        labels_dict = results["labels"]
        if labels_dict is None:
            return docs

        for head_name, labels in labels_dict.items():
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            for doc, label in zip(docs, labels):
                if head_name in self.id2label and isinstance(label, int):
                    label = self.id2label[head_name].get(label, label)
                setattr(doc._, head_name, label)

        return docs

    def to_disk(self, path, *, exclude=set()):
        """
        Save the classifier state to disk.

        Stores:
        - Label attributes and mappings
        - Per-head configuration (loss, hidden size, dropout, etc.)

        Parameters
        ----------
        path : Path
            Directory where files are saved.
        exclude : set, optional
            Components to exclude from saving.
        """
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        exclude.add(repr_id)
        os.makedirs(path, exist_ok=True)
        data_path = path / "multi_head_data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "label_attr": self.label_attr,
                    "label2id": self.label2id,
                    "id2label": self.id2label,
                    "loss_config": self.loss_config,
                    "hidden_size_config": self.hidden_size_config,
                    "activation_mode_config": self.activation_mode_config,
                    "dropout_rate_config": self.dropout_rate_config,
                    "layer_norm_config": self.layer_norm_config,
                },
                f,
            )
        return super().to_disk(path, exclude=exclude)

    @classmethod
    def from_disk(cls, path, **kwargs):
        """
        Load a classifier from disk.

        Restores label mappings, per-head configurations, and rebuilds
        the classifier architecture.

        Parameters
        ----------
        path : Path
            Directory containing saved files.
        kwargs : dict
            Extra arguments passed to the constructor.

        Returns
        -------
        TrainableDocClassifier
            Restored classifier instance.
        """
        data_path = path / "multi_head_data.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        obj = super().from_disk(path, **kwargs)
        obj.label_attr = data.get("label_attr", [])
        obj.head_names = obj.label_attr
        obj.label2id = data.get("label2id", {})
        obj.id2label = data.get("id2label", {})
        obj.loss_config = data.get("loss_config", {})
        obj.hidden_size_config = data.get("hidden_size_config", {})
        obj.activation_mode_config = data.get("activation_mode_config", {})
        obj.dropout_rate_config = data.get("dropout_rate_config", {})
        obj.layer_norm_config = data.get("layer_norm_config", {})
        return obj
