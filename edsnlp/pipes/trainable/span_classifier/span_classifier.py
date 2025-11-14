from __future__ import annotations

import logging
import os
import pickle
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import foldedtensor as ft
import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import NotRequired, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.base import BaseSpanAttributeClassifierComponent
from edsnlp.pipes.trainable.embeddings.typing import (
    SpanEmbeddingComponent,
)
from edsnlp.pipes.trainable.span_classifier.lrt import lrt_flip_scheme
from edsnlp.utils.bindings import (
    BINDING_GETTERS,
    BINDING_SETTERS,
    Attributes,
    AttributesArg,
)
from edsnlp.utils.span_getters import (
    ContextWindow,
    SpanFilter,
    SpanGetterArg,
    get_spans,
)

logger = logging.getLogger(__name__)

SpanClassifierBatchInput = TypedDict(
    "SpanClassifierBatchInput",
    {
        "embedding": BatchInput,
        "targets": NotRequired[torch.Tensor],
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
mask: torch.BoolTensor
    Mask of the sequences
spans: torch.Tensor
    2d tensor of n_spans * (doc_idx, ner_label_idx, begin, end)
targets: NotRequired[List[torch.Tensor]]
    list of 2d tensor of n_spans * n_combinations (1 hot)
"""

SpanClassifierBatchOutput = TypedDict(
    "SpanClassifierBatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "labels": Optional[List[torch.Tensor]],
    },
)
"""
loss: Optional[torch.Tensor]
    The loss of the model
labels: Optional[List[torch.Tensor]]
    The predicted labels
"""


class TrainableSpanClassifier(
    TorchComponent[BatchOutput, SpanClassifierBatchInput],
    BaseSpanAttributeClassifierComponent,
):
    """
    The `eds.span_classifier` component is a trainable attribute predictor.
    In this context, the span classification task consists in assigning values (boolean,
    strings or any object) to attributes/extensions of spans such as:

    - `span._.negation`,
    - `span._.date.mode`
    - `span._.cui`

    In the rest of this page, we will refer to a pair of (attribute, value) as a
    "binding". For instance, the binding `("_.negation", True)` means that the
    attribute `negation` of the span is (or should be, when predicted) set to `True`.

    Architecture
    ------------
    The model performs span classification by:

    1. Calling a word pooling embedding such as `eds.span_pooler` to compute a single
    embedding for each span
    2. Computing logits for each possible binding using a linear layer
    3. Splitting these bindings into groups of exclusive values such as

        - `event=start` and `event=stop`
        - `negated=False` and `negated=True`

        Note that the above groups are not exclusive, but the values within each group
        are.

    4. Applying the best scoring binding in each group to each span

    Examples
    --------
    To create a span classifier component, you can use the following code:

    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_classifier(
            # To embed the spans, we will use a span pooler
            embedding=eds.span_pooler(
                pooling_mode="mean",  # mean pooling
                # that will use a transformer to embed the doc words
                embedding=eds.transformer(
                    model="prajjwal1/bert-tiny",
                    window=128,
                    stride=96,
                ),
            ),
            span_getter=["ents", "sc"],
            # For every span embedded by the span pooler
            # (doc.ents and doc.spans["sc"]), we will predict both
            # span._.negation and span._.event_type
            attributes=["_.negation", "_.event_type"],
        ),
        name="span_classifier",
    )
    ```

    To infer the values of the attributes, you can use the pipeline `post_init` method:

    ```{ .python .no-check }
    nlp.post_init(gold_data)
    ```

    To train the model, refer to the [Training](/tutorials/training-span-classifier)
    tutorial.

    You can inspect the bindings that will be used for training and prediction
    ```{ .python .no-check }
    print(nlp.pipes.attr.bindings)
    # list of (attr name, span labels or True if all, values)
    # Out: [
    #   ('_.negation', True, [True, False]),
    #   ('_.event_type', True, ['start', 'stop'])
    # ]
    ```

    You can also change these values and update the bindings by calling the
    `update_bindings` method. Don't forget to retrain the model if new values are
    added !

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    embedding : SpanEmbeddingComponent
        The word embedding component
    label_weights: Dict[str, Dict[Any, float]]
        The weight of each label for each attribute. The keys are the attribute names
        and the values are dictionaries with the labels as keys and the weights as
        values. For instance, `{"_.negation": {True: 1, False: 2}}` will give a weight
        of 1 to the `True` value of the `negation` attribute and 2 to the `False` value.
    span_getter : SpanGetterArg
        How to extract the candidate spans and the attributes to predict or train on.
    context_getter : Optional[Union[Callable, SpanGetterArg]]
        What context to use when computing the span embeddings (defaults to the whole
        document). This can be:

        - a `SpanGetterArg` to retrieve contexts from a whole document. For example
          `{"section": "conclusion"}` to only use the conclusion as context (you
          must ensure that all spans produced by the `span_getter` argument do fall
          in the conclusion in this case)
        - a callable, that gets a span and should return a context for this span.
          For instance, `lambda span: span.sent` to use the sentence as context.
    attributes : AttributesArg
        The attributes to predict or train on. If a dict is given, keys are the
        attributes and values are the labels for which the attr is allowed, or True
        if the attr is allowed for all labels.
    keep_none : bool
        If False, skip spans for which a attr returns None. If True (default), the
        None values will be learned and predicted, just as any other value.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "span_classifier",
        *,
        embedding: SpanEmbeddingComponent,
        attributes: AttributesArg = None,
        qualifiers: AttributesArg = None,
        label_weights: Dict[str, Dict[Any, float]] = None,
        span_getter: SpanGetterArg = None,
        context_getter: Optional[Union[ContextWindow, SpanGetterArg]] = None,
        values: Optional[Dict[str, List[Any]]] = None,
        keep_none: bool = False,
        loss_fn: Optional[Callable] = None,
        deduplicate: bool = False,
    ):
        attributes: Attributes
        if attributes is None and qualifiers is None:
            raise TypeError(
                "The `attributes` parameter is required. Please provide a dict of "
                "attributes to predict or train on."
            )

        if qualifiers is not None:
            warnings.warn(
                "The `qualifiers` parameter is deprecated. Use `attributes` instead."
            )
            assert attributes is None
            attributes = qualifiers
        sub_span_getter = getattr(embedding, "span_getter", None)
        if (
            sub_span_getter is not None and span_getter is None
        ):  # pragma: no cover  # noqa: E501
            span_getter = sub_span_getter
        span_getter = span_getter or {"ents": True}
        sub_context_getter = getattr(embedding, "context_getter", None)
        if (
            sub_context_getter is not None and context_getter is None
        ):  # pragma: no cover
            context_getter = sub_context_getter

        self.values = values
        self.keep_none = keep_none

        attributes = {
            k if k.startswith("_.") else f"_.{k}": v for k, v in attributes.items()
        }
        self.label_weights_bindings = (
            {
                (attr if attr.startswith("_.") else f"_.{attr}", value): weight
                for attr, values in label_weights.items()
                for value, weight in values.items()
            }
            if label_weights
            else dict()
        )

        unknown = set([attr for attr, _ in self.label_weights_bindings]) - set(
            attributes.keys()
        )
        if unknown:
            warnings.warn(
                f"Attributes ({unknown}) are present in label_weights "
                f"but not in attributes: Those weights will be ignored"
            )

        self.bindings: List[Tuple[str, List[str], List[Any]]] = [
            (k, v, []) for k, v in attributes.items()
        ]

        super().__init__(nlp, name, span_getter=span_getter)
        self.embedding = embedding
        self.context_getter = context_getter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.classifier = torch.nn.Linear(embedding.output_size, 0)

        self.loss_fn = loss_fn
        self.deduplicate = deduplicate
        self.corrected_targets = {}
        for b in self.bindings:
            self.corrected_targets[b[0]] = {}

    @property
    def attributes(self) -> Attributes:
        return {qlf: labels for qlf, labels, _ in self.bindings}

    @attributes.setter
    def attributes(self, value: Attributes):
        bindings = []
        for qlf, labels in value.items():
            groups = [group for group in self.bindings if group[0] == qlf]
            if len(groups) > 1:
                raise ValueError(
                    f"Attribute {qlf} has different label filters: "
                    f"{[g[0] for g in groups]}. Please use the `update_bindings` "
                    f"method to update the labels."
                )
            if groups:
                bindings.append((qlf, labels, groups[0][2]))
        self.update_bindings(bindings)

    def to_disk(self, path, *, exclude=set()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        # We save the bindings as a pickle file since values can be arbitrary objects
        os.makedirs(path, exist_ok=True)
        data_path = path / "bindings.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "bindings": self.bindings,
                },
                f,
            )
        return super().to_disk(path, exclude=exclude)

    def from_disk(self, path, exclude=tuple()):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        # This will receive the directory path + /my_component
        data_path = path / "bindings.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.update_bindings(data["bindings"])
        self.set_extensions()
        super().from_disk(path, exclude=exclude)

    @property
    def bindings_to_idx(self) -> List[Tuple[str, List[str], Dict[Any, int]]]:
        if getattr(self, "_bindings_to_idx", None) is not None:
            return self._bindings_to_idx
        self._bindings_to_idx = [  # noqa
            (attr, labels, {value: idx for idx, value in enumerate(values)})
            for attr, labels, values in self.bindings
        ]
        return self._bindings_to_idx

    @property
    def bindings_indexers(self) -> List[Union[Sequence[int], slice]]:
        return self._bindings_indexers

    def set_extensions(self):
        super().set_extensions()
        for group in self.bindings:
            qlf = group[0]
            if qlf.startswith("_."):
                qlf = qlf[2:]
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def post_init(
        self,
        gold_data: Iterable[Doc],
        exclude: Set[str],
    ):
        super().post_init(gold_data, exclude=exclude)

        bindings = [
            (qlf, labels, dict.fromkeys(vals)) for qlf, labels, vals in self.bindings
        ]
        self.binding_target_shape = dict.fromkeys(qlf for qlf, _, _ in bindings)

        for doc in gold_data:
            spans = list(get_spans(doc, self.span_getter, deduplicate=self.deduplicate))
            for span in spans:
                for attr, labels, values in bindings:  # FIXME
                    binding_has_softlabels = False
                    if labels is True or span.label_ in labels:
                        value = BINDING_GETTERS[attr](span)
                        if value is not None or self.keep_none:
                            if isinstance(value, dict):
                                binding_has_softlabels = True
                                for k in value.keys():
                                    values[k] = None
                            else:
                                values[value] = None

                        self.binding_target_shape[attr] = (
                            len(values) if binding_has_softlabels else 1
                        )

        bindings = [
            (attr, labels, sorted(values, key=str)) for attr, labels, values in bindings
        ]

        for attr, labels, values in bindings:
            if len(values) < 2:
                warnings.warn(
                    f"Attribute {attr} for labels {labels} should have at "
                    f"least 2 values but found {len(values)}: {values}."
                )
        self.exist_soft_labels = max(self.binding_target_shape.values()) > 1
        self.update_bindings(bindings)

    def update_bindings(self, bindings: List[Tuple[str, SpanFilter, List[Any]]]):
        dev = self.classifier.weight.device
        keep_bindings = [
            (labels is True or len(labels) > 0) and len(values) > 0
            for k, labels, values in bindings
        ]
        if not all(keep_bindings):
            logger.warning(
                "Some attributes have no labels or values and have been removed:"
                + "".join(
                    "\n- " + str(b[0]) + " for labels " + str(b[1])
                    for b, keep in zip(bindings, keep_bindings)
                    if not keep
                )
            )
        bindings = [b for b, keep in zip(bindings, keep_bindings) if keep]
        new_bindings = list(
            dict.fromkeys((k, v) for k, _, group in bindings for v in group)
        )
        old_bindings = list(
            dict.fromkeys((k, v) for k, _, group in self.bindings for v in group)
        )

        new_bindings_to_idx = {binding: idx for idx, binding in enumerate(new_bindings)}
        old_bindings_to_idx = {binding: idx for idx, binding in enumerate(old_bindings)}

        common = [b for b in new_bindings if b in old_bindings]
        old_index = [old_bindings_to_idx[label] for label in common]
        new_index = [new_bindings_to_idx[label] for label in common]
        new_linear = torch.nn.Linear(self.classifier.in_features, len(new_bindings)).to(
            dev
        )
        new_linear.weight.data[new_index] = self.classifier.weight.data[old_index]
        self.classifier.weight.data = new_linear.weight.data
        self.classifier.out_features = new_bindings
        missing_bindings = set(new_bindings) - set(old_bindings)
        if missing_bindings and len(old_bindings) > 0:
            warnings.warn(
                f"Added {len(missing_bindings)} new bindings. Consider retraining "
                f"the model to learn these new bindings."
            )

        if hasattr(self.classifier, "bias"):
            new_linear.bias.data[new_index] = self.classifier.bias.data[old_index]
            self.classifier.bias.data = new_linear.bias.data

        def simplify_indexer(indexer):
            return (
                slice(indexer[0], indexer[-1] + 1)
                if len(indexer) and list(range(indexer[0], indexer[-1] + 1)) == indexer
                else indexer
            )

        self.bindings = bindings
        self._bindings_indexers = [
            simplify_indexer([new_bindings_to_idx[(qlf, value)] for value in values])
            for qlf, labels, values in bindings
        ]
        self.label_weights = [
            self.label_weights_bindings.get((qlf, value), 1)
            for qlf, labels, values in bindings
            for value in values
        ]
        self._bindings_to_idx = None

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        spans = list(get_spans(doc, self.span_getter, deduplicate=self.deduplicate))
        span_ids = [span._.instance_id for span in spans]
        if self.context_getter is None or not callable(self.context_getter):
            contexts = list(
                get_spans(doc, self.context_getter, deduplicate=self.deduplicate)
            )
            pre_aligned = False
        else:
            contexts = [self.context_getter(span) for span in spans]
            pre_aligned = True
        return {
            "embedding": self.embedding.preprocess(
                doc,
                spans=spans,
                contexts=contexts,
                pre_aligned=pre_aligned,
                **kwargs,
            ),
            "$spans": spans,
            "span_ids": span_ids,
        }

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        preps = self.preprocess(doc)
        targets = []
        for span in preps["$spans"]:
            span_targets = []
            for qlf, labels, values_to_idx in self.bindings_to_idx:
                if labels is True or span.label_ in labels:
                    value = BINDING_GETTERS[qlf](span)
                    if isinstance(value, dict):
                        # Probabilities dict: convert to vector in
                        # order of values_to_idx
                        prob_vec = [
                            float(value.get(val, -100)) for val in values_to_idx
                        ]

                        span_targets.append(prob_vec)
                    else:
                        idx = values_to_idx.get(value, -100)
                        if self.exist_soft_labels:
                            if idx != -100:
                                target = F.one_hot(
                                    torch.tensor(idx),
                                    num_classes=len(values_to_idx),
                                ).tolist()
                            else:
                                target = [idx] * len(values_to_idx)
                        else:
                            target = idx
                        span_targets.append(target)
                else:
                    if self.exist_soft_labels:
                        ignore_value = [-100] * len(values_to_idx)
                        span_targets.append(ignore_value)
                    else:
                        span_targets.append(-100)
            targets.append(span_targets)
        return {**preps, "targets": targets}

    # def transform_label(self, value, values_to_idx, target_shape):
    #     if isinstance(value, dict):
    #         target = [float(value.get(val, -100)) for val in values_to_idx]
    #     else:
    #         idx = values_to_idx.get(value, -100)
    #         if target_shape > 1:
    #             target = F.one_hot(
    #                 torch.tensor(idx),
    #                 num_classes=len(values_to_idx),
    #             ).tolist()
    #         else:
    #             target = idx
    #     return target

    # def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
    #     preps = self.preprocess(doc)
    #     targets = []

    #     for qlf, labels, values_to_idx in self.bindings_to_idx:
    #         qlf_target_shape = self.binding_target_shape[qlf]
    #         qlf_targets = []
    #         for span in preps["$spans"]:
    #             if labels is True or span.label_ in labels:
    #                 value = BINDING_GETTERS[qlf](span)
    #                 target = self.transform_label(
    #                     value, values_to_idx, qlf_target_shape
    #                 )
    #             else:
    #                 if qlf_target_shape > 1:
    #                     target = [-100] * len(values_to_idx)
    #                 else:
    #                     target = -100
    #             qlf_targets.append(target)
    #         targets.append(qlf_targets)
    #     return {**preps, "targets": targets}

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanClassifierBatchInput:
        collated: SpanClassifierBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
        }
        if "targets" in batch:
            targets = ft.as_folded_tensor(
                batch["targets"],
                dtype=torch.float if self.exist_soft_labels else torch.long,
                full_names=("sample", "span", "group"),
                data_dims=("span", "group"),
            ).as_tensor()
            if self.exist_soft_labels:
                collated["targets"] = targets
            else:
                collated["targets"] = targets.view(len(targets), len(self.bindings))
        if "span_ids" in batch:
            collated["span_ids"] = torch.tensor(batch["span_ids"], dtype=torch.long)

        return collated

    def correct_targets(self, batch, group_idx, mask, targets):
        # If external corrected targets were provided, override (coalesce)
        # the corresponding rows in `targets` for spans whose instance ids
        # appear in corrected_targets for this task.
        if "span_ids" in batch:
            task = self.bindings[group_idx][0]
            span_ids = batch["span_ids"].squeeze(0)[mask]
            # corrected_targets expected shape:
            # { "_.attr": { span_instance_id: { value: prob, ... }  OR single_value } }
            corr_for_task = self.corrected_targets.get(task, {})
            if corr_for_task and span_ids.numel() > 0:
                # targets may be float (soft labels) or long (discrete)
                # make a mutable copy
                targets = targets.clone()
                # values ordering for this group
                _, _, values_order = self.bindings[group_idx]
                # dict that maps value->idx for this group
                _, _, values_to_idx = self.bindings_to_idx[group_idx]
                for row_idx, sid in enumerate(span_ids.tolist()):
                    if sid in corr_for_task:
                        corr_val = corr_for_task[sid]
                        # Soft distribution provided as dict
                        if isinstance(corr_val, dict):
                            vec = torch.tensor(
                                [
                                    float(corr_val.get(str(v), -100))
                                    for v in values_order
                                ],
                                dtype=torch.float,
                                device=targets.device,
                            )
                            if self.exist_soft_labels:
                                targets[row_idx] = vec
                            else:
                                # convert distribution to discrete index (argmax),
                                #  unless all -100
                                if torch.all(vec == -100):
                                    targets[row_idx] = -100
                                else:
                                    targets[row_idx] = int(torch.argmax(vec).item())
                        else:
                            # Single discrete value provided
                            idx = values_to_idx.get(corr_val, -100)
                            if self.exist_soft_labels:
                                if idx != -100:
                                    onehot = F.one_hot(
                                        torch.tensor(idx, device=targets.device),
                                        num_classes=len(values_order),
                                    ).to(torch.float)
                                    targets[row_idx] = onehot
                                else:
                                    targets[row_idx] = torch.tensor(
                                        [-100.0] * len(values_order),
                                        device=targets.device,
                                    )
                            else:
                                targets[row_idx] = idx
                # replace `targets` used below with the coalesced version
                # (preds and loss computation will use this variable)
                # no further action required here

        return targets

    def lrt_scheme(self, scores, targets, lrt_parameters):
        new_y_tilde, changed_idx = lrt_flip_scheme(
            scores, targets, delta=lrt_parameters.get("delta", 0.5)
        )
        return new_y_tilde, changed_idx

    def update_corrected_targets(self, group_idx, new_y_tilde, changed_idx, span_ids):
        task = self.bindings[group_idx][0]
        task_labels = self.bindings[group_idx][2]
        for idx in changed_idx:
            idx = idx.item()
            span_id = span_ids[idx].item()
            self.corrected_targets[task][span_id] = {
                k: v.item()
                for k, v in zip(
                    task_labels,
                    torch.nn.functional.one_hot(
                        new_y_tilde[idx], num_classes=len(task_labels)
                    ),
                )
            }

        logger.debug(f"## {len(changed_idx)} data points changed ##")

    # noinspection SpellCheckingInspection
    def forward(
        self,
        batch: SpanClassifierBatchInput,
        apply_lrt: bool = False,
        use_corrected_targets: bool = False,
        lrt_parameters: Optional[Dict[str, Any]] = {},
    ) -> BatchOutput:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans

        Parameters
        ----------
        batch: SpanClassifierBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        embedding = self.embedding(batch["embedding"])
        span_embeds = embedding["embeddings"]

        binding_scores = self.classifier(span_embeds)

        if "targets" in batch:
            losses = []
            span_ids = None
        else:
            predictions = []
            losses = None
            scores = None
            span_ids = None
            targets = None

        # For each group, for instance:
        # - `event=start` and `event=stop`
        # - `negated=False` and `negated=True`
        for group_idx, bindings_indexer in enumerate(self.bindings_indexers):
            if "targets" in batch:
                if self.exist_soft_labels:
                    mask = torch.all(batch["targets"][:, group_idx] != -100, axis=1)
                else:
                    mask = batch["targets"][:, group_idx] != -100
                logits = binding_scores[mask, bindings_indexer]
                targets = batch["targets"][mask, group_idx]

                scores = torch.softmax(
                    logits.as_tensor().detach(), dim=1
                )  # FIXME append to a list , else we are overwriting
                predictions = logits.argmax(
                    dim=1
                ).as_tensor()  # FIXME append to a list , else we are overwriting

                if "span_ids" in batch:
                    span_ids = batch["span_ids"].squeeze(0)

                ## Apply here the target correction
                if apply_lrt:
                    new_y_tilde, changed_idx = self.lrt_scheme(
                        scores, targets, lrt_parameters=lrt_parameters
                    )
                    self.update_corrected_targets(
                        group_idx, new_y_tilde, changed_idx, span_ids
                    )
                if use_corrected_targets:
                    targets = self.correct_targets(batch, group_idx, mask, targets)

                # Loss computation
                if self.loss_fn is not None:
                    loss = self.loss_fn(logits, targets, **{"model": self})
                else:
                    loss = F.cross_entropy(
                        logits,
                        targets,
                        reduction="sum",
                        weight=torch.tensor(self.label_weights, dtype=torch.float)[
                            bindings_indexer
                        ].to(binding_scores.device),
                    )
                losses.append(loss)
                assert not torch.isnan(losses[-1]).any(), "NaN loss"
            else:
                predictions.append(binding_scores[:, bindings_indexer].argmax(dim=1))

        return {
            "loss": sum(losses) if losses is not None else None,
            "labels": predictions,
            "scores": scores,
            "span_ids": span_ids,
            "targets": targets,
        }

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: SpanClassifierBatchOutput,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[Doc]:
        # Preprocessed docs should still be in the cache
        spans = [span for sample in inputs for span in sample["$spans"]]
        all_labels = results["labels"]
        # For each prediction group (exclusive bindings)...
        for val_indices, (qlf, labels, values) in zip(all_labels, self.bindings):
            # For each span...
            for span, idx in zip(spans, val_indices.tolist()):
                # If the span is not filtered out...
                if labels is True or span.label_ in labels:
                    # ...assign the predicted value to the span
                    BINDING_SETTERS[qlf](span, values[idx])
        return docs


# For backward compatibility
TrainableSpanQualifier = TrainableSpanClassifier
