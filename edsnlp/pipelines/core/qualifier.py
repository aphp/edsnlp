from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any, Union

from spacy.scorer import PRFScore
from thinc.types import Floats2d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens import Doc, Span
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer


Span.set_extension("qualifiers", default={}, force=True)
msg = Printer()


@Language.factory(
    "qualifier",
    requires=["doc.ents"],
    assigns=["span._.qualifiers"],
    default_score_weights={
        "qual_micro_p": 0.0,
        "qual_micro_r": 0.0,
        "qual_micro_f": 1.0,
    },
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    model: Model,
    *,
    threshold: float,
):
    """Construct a TrainableQualifier component."""
    return TrainableQualifier(nlp.vocab, model, name, threshold=threshold)


class TrainableQualifier(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "qualifier",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError(
                "Only strings can be added as labels to the RelationExtractor"
            )
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(doc.ents)
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        total_instances = sum([len(doc.ents) for doc in docs])
        if total_instances == 0:
            msg.info(
                "Could not determine any instances in any docs - can not make any predictions."
            )
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        for doc in docs:
            for ent in doc.ents:
                for j, label in enumerate(self.labels):
                    ent._.qualifiers[label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(eg.predicted.ents)
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths, mask = self._examples_to_truth(examples)
        gradient = (scores - truths) * mask
        mean_square_error = (gradient**2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                for ent in example.reference.ents:
                    for label in ent._qualifiers.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError(
                "Call begin_training with relevant entities and relations annotated in "
                "at least a few reference examples!"
            )
        self.model.initialize(X=doc_sample, Y=label_sample[0])

    def _examples_to_truth(
        self, examples: List[Example]
    ) -> Optional[Tuple[numpy.ndarray, numpy.ndarray]]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(eg.reference.ents)
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        mask = numpy.ones((nr_instances, len(self.labels)), dtype="f")

        c = 0
        for eg in examples:
            for ent in eg.reference.ents:
                qualifiers = prepare_qualifiers(ent._.qualifiers, self.labels)
                for i, label in enumerate(self.labels):
                    if label not in qualifiers:
                        mask[c, i] = 0
                    else:
                        truths[c, i] = float(qualifiers[label])
                c += 1

        truths = self.model.ops.asarray(truths)
        mask = self.model.ops.asarray(mask)
        return truths, mask

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return score_qualifiers(examples, self.threshold)


def prepare_qualifiers(
    qualifiers: Union[List[str], Dict[str, float]],
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    if isinstance(qualifiers, dict):
        return qualifiers
    elif labels is not None:
        qualifiers = {label: 1.0 if label in qualifiers else 0.0 for label in labels}
    else:
        qualifiers = {k: 1.0 for k in qualifiers}
    return qualifiers


def score_qualifiers(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""
    micro_prf = PRFScore()
    for example in examples:
        gold = example.reference._.qualifiers
        pred = example.predicted._.qualifiers

        gold = prepare_qualifiers(gold)

        gold_labels = [k for (k, v) in gold.items() if v == 1.0]

        for k, v in pred.items():
            if v >= threshold:
                if k in gold_labels:
                    micro_prf.tp += 1
                else:
                    micro_prf.fp += 1
            else:
                if k in gold_labels:
                    micro_prf.fn += 1
    return {
        "qual_micro_p": micro_prf.precision,
        "qual_micro_r": micro_prf.recall,
        "qual_micro_f": micro_prf.fscore,
    }
