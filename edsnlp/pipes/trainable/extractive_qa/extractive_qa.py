from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp.core.pipeline import Pipeline
from edsnlp.pipes.trainable.embeddings.typing import (
    WordEmbeddingComponent,
)
from edsnlp.pipes.trainable.ner_crf.ner_crf import NERBatchOutput, TrainableNerCrf
from edsnlp.utils.filter import align_spans, filter_spans
from edsnlp.utils.span_getters import (
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
)
from edsnlp.utils.typing import AsList


class TrainableExtractiveQA(TrainableNerCrf):
    """
    The `eds.extractive_qa` component is a trainable extractive question answering
    component. This can be seen as a Named Entity Recognition (NER) component where the
    types of entities predicted by the model are not pre-defined during the training
    but are provided as prompts (i.e., questions) at inference time.

    The `eds.extractive_qa` shares a lot of similarities with the `eds.ner_crf`
    component, and therefore most of the arguments are the same.

    !!! note "Extractive vs Abstractive Question Answering"

        Extractive Question Answering differs from Abstractive Question Answering in
        that the answer is extracted from the text, rather than generated (à la
        ChatGPT) from scratch. To normalize the answers, you can use the
        `eds.span_linker` component in `synonym` mode and search for the closest
        `synonym` in a predefined list.

    Examples
    --------

    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.extractive_qa(
            embedding=eds.transformer(
                model="prajjwal1/bert-tiny",
                window=128,
                stride=96,
            ),
            mode="joint",
            target_span_getter="ner-gold",
            span_setter="ents",
            questions={
                "disease": "What disease does the patient have?",
                "drug": "What drug is the patient taking?",
            },  # (1)!
        ),
        name="qa",
    )
    ```

    To train the model, refer to the [Training](/tutorials/make-a-training-script)
    tutorial.

    Once the model is trained, you can use the questions attribute (next section) on the
    document you run the model on, or you can change the global questions attribute:

    ```python
    nlp.pipes.qa.questions = {
        "disease": "When did the patient get sick?",
    }
    ```

    # Dynamic Questions

    You can also provide

    ```{ .python .no-check }
    eds.extractive_qa(..., questions_attribute="questions")
    ```

    to get the questions dynamically from an attribute on the Doc or Span objects
    (e.g., `doc._.questions`). This is useful when you want to have different questions
    depending on the document.

    To provide questions from a dataframe, you can use the following code:

    ```{ .python .no-check }
    dataframe = pd.DataFrame({"questions": ..., "note_text": ..., "note_id": ...})
    stream = edsnlp.data.from_pandas(
        dataframe,
        converter="omop",
        doc_attributes={"questions": "questions"},
    )
    stream.map_pipeline(nlp)
    stream.set_processing(backend="multiprocessing")
    out = stream.to_pandas(converters="ents")
    ```


    Parameters
    ----------
    name : str
        Name of the component
    embedding : WordEmbeddingComponent
        The word embedding component
    questions : Dict[str, AsList[str]]
        The questions to ask, as a mapping between the entity type and the list of
        questions to ask for this entity type (or single string if only one question).
    questions_attribute : Optional[str]
        The attribute to use to get the questions dynamically from the Doc or Span
        objects (as returned by the `context_getter` argument). If None, the questions
        will be fixed and only taken from the `questions` argument.
    context_getter : Optional[SpanGetterArg]
        What context to use when computing the span embeddings (defaults to the whole
        document). For example `{"section": "conclusion"}` to only extract the
        entities from the conclusion.
    target_span_getter : SpanGetterArg
        Method to call to get the gold spans from a document, for scoring or training.
        By default, takes all entities in `doc.ents`, but we recommend you specify
        a given span group name instead.
    span_setter : Optional[SpanSetterArg]
        The span setter to use to set the predicted spans on the Doc object. If None,
        the component will infer the span setter from the target_span_getter config.
    infer_span_setter : Optional[bool]
        Whether to complete the span setter from the target_span_getter config.
        False by default, unless the span_setter is None.
    mode : Literal["independent", "joint", "marginal"]
        The CRF mode to use : independent, joint or marginal
    window : int
        The window size to use for the CRF. If 0, will use the whole document, at
        the cost of a longer computation time. If 1, this is equivalent to assuming
        that the tags are independent and will the component be faster, but with
        degraded performance. Empirically, we found that a window size of 10 or 20
        works well.
    stride : Optional[int]
        The stride to use for the CRF windows. Defaults to `window // 2`.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: Optional[str] = "extractive_qa",
        *,
        embedding: WordEmbeddingComponent,
        questions: Dict[str, AsList[str]] = {},
        questions_attribute: str = "questions",
        context_getter: Optional[SpanGetterArg] = None,
        target_span_getter: Optional[SpanGetterArg] = None,
        span_setter: Optional[SpanSetterArg] = None,
        infer_span_setter: Optional[bool] = None,
        mode: Literal["independent", "joint", "marginal"] = "joint",
        window: int = 40,
        stride: Optional[int] = None,
    ):
        self.questions_attribute: Optional[str] = questions_attribute
        self.questions = questions
        super().__init__(
            nlp=nlp,
            name=name,
            embedding=embedding,
            context_getter=context_getter,
            span_setter=span_setter,
            target_span_getter=target_span_getter,
            mode=mode,
            window=window,
            stride=stride,
            infer_span_setter=infer_span_setter,
        )
        self.update_labels(["answer"])
        self.labels_to_idx = defaultdict(lambda: 0)

    def set_extensions(self):
        super().set_extensions()
        if self.questions_attribute:
            if not Doc.has_extension(self.questions_attribute):
                Doc.set_extension(self.questions_attribute, default=None)
            if not Span.has_extension(self.questions_attribute):
                Span.set_extension(self.questions_attribute, default=None)

    def post_init(self, docs: Iterable[Doc], exclude: Set[str]):
        pass

    @property
    def cfg(self):
        cfg = dict(super().cfg)
        cfg.pop("labels")
        return cfg

    def preprocess(self, doc, **kwargs):
        contexts = (
            list(get_spans(doc, self.context_getter))
            if self.context_getter
            else [doc[:]]
        )
        prompt_contexts_and_labels = sorted(
            {
                (prompt, label, context)
                for context in contexts
                for label, questions in (
                    *self.questions.items(),
                    *(getattr(doc._, self.questions_attribute) or {}).items(),
                    *(
                        (getattr(context._, self.questions_attribute) or {}).items()
                        if context is not doc
                        else ()
                    ),
                )
                for prompt in questions
            }
        )
        questions = [x[0] for x in prompt_contexts_and_labels]
        labels = [x[1] for x in prompt_contexts_and_labels]
        ctxs = [x[2] for x in prompt_contexts_and_labels]
        lengths = [len(ctx) for ctx in ctxs]
        return {
            "lengths": lengths,
            "$labels": labels,
            "$contexts": ctxs,
            "embedding": self.embedding.preprocess(
                doc,
                contexts=ctxs,
                prompts=questions,
                **kwargs,
            ),
            "stats": {"ner_words": sum(lengths)},
        }

    def preprocess_supervised(self, doc, **kwargs):
        prep = self.preprocess(doc, **kwargs)
        contexts = prep["$contexts"]
        labels = prep["$labels"]
        tags = []

        for context, label, target_ents in zip(
            contexts,
            labels,
            align_spans(
                list(get_spans(doc, self.target_span_getter)),
                contexts,
            ),
        ):
            span_tags = [[0] * len(self.labels) for _ in range(len(context))]
            start = context.start
            target_ents = [ent for ent in target_ents if ent.label_ == label]

            # TODO: move this to the LinearChainCRF class
            for ent in filter_spans(target_ents):
                label_idx = self.labels_to_idx[ent.label_]
                if ent.start == ent.end - 1:
                    span_tags[ent.start - start][label_idx] = 4
                else:
                    span_tags[ent.start - start][label_idx] = 2
                    span_tags[ent.end - 1 - start][label_idx] = 3
                    for i in range(ent.start + 1 - start, ent.end - 1 - start):
                        span_tags[i][label_idx] = 1
            tags.append(span_tags)

        return {
            **prep,
            "targets": tags,
        }

    def postprocess(
        self,
        docs: List[Doc],
        results: NERBatchOutput,
        inputs: List[Dict[str, Any]],
    ):
        spans: Dict[Doc, list[Span]] = defaultdict(list)
        contexts = [ctx for sample in inputs for ctx in sample["$contexts"]]
        labels = [label for sample in inputs for label in sample["$labels"]]
        tags = results["tags"].cpu()
        for context_idx, _, start, end in self.crf.tags_to_spans(tags).tolist():
            span = contexts[context_idx][start:end]
            label = labels[context_idx]
            span.label_ = label
            spans[span.doc].append(span)
        for doc in docs:
            self.set_spans(doc, spans.get(doc, []))
        return docs
