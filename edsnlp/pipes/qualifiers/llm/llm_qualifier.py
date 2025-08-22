from __future__ import annotations

import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from spacy.tokens import Doc, Span
from typing_extensions import TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.pipes.base import BaseSpanAttributeClassifierComponent
from edsnlp.pipes.qualifiers.llm.llm_utils import (
    AsyncLLM,
    create_prompt_messages,
)
from edsnlp.utils.asynchronous import run_async
from edsnlp.utils.bindings import (
    BINDING_SETTERS,
    Attributes,
    AttributesArg,
)
from edsnlp.utils.span_getters import SpanGetterArg, get_spans

logger = logging.getLogger(__name__)

LLMSpanClassifierBatchInput = TypedDict(
    "LLMSpanClassifierBatchInput",
    {
        "queries": List[str],
    },
)
"""
queries: List[str]
    List of queries to send to the LLM for classification.
    Each query corresponds to a span and its context.
"""

LLMSpanClassifierBatchOutput = TypedDict(
    "LLMSpanClassifierBatchOutput",
    {
        "labels": Optional[Union[List[str], List[List[str]]]],
    },
)
"""
labels: Optional[Union[List[str], List[List[str]]]]
    The predicted labels for each query.
    If `n > 1`, this will be a list of lists, where each inner list contains the
    predictions for a single query.
    If `n == 1`, this will be a list of strings, where each string is the prediction
    for a single query.

    If the API call fails or no predictions are made, this will be None.
    If `n > 1`, it will be a list of None values for each query.
    If `n == 1`, it will be a single None value.

"""


class LLMSpanClassifier(
    BaseSpanAttributeClassifierComponent,
):
    """
    The `LLMSpanClassifier` component is a LLM attribute predictor.
    In this context, the span classification task consists in assigning values (boolean,
    strings or any object) to attributes/extensions of spans such as:

    - `span._.negation`,
    - `span._.date.mode`
    - `span._.cui`

    In the rest of this page, we will refer to a pair of (attribute, value) as a
    "binding". For instance, the binding `("_.negation", True)` means that the
    attribute `negation` of the span is (or should be, when predicted) set to `True`.

    Python >= '3.8' is required to use this pipeline.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    system_prompt : Optional[str]
        A system prompt to use for the LLM. This is a general prompt that will be
        prepended to each query. This prompt will be passed under the `system` role
        in the OpenAI API call.
        Example: "You are a medical expert. Classify the following text."
        If None, no system prompt will be used.
        Note: This is not the same as the `user_prompt` parameter.
    user_prompt : Optional[str]
        A general prompt to use for all spans. This is a prompt that will be prepended
        to each span's specific prompt. This will be passed under the `user` role
        in the OpenAI API call.
    prefix_prompt : Optional[str]
        A prefix prompt to paste after the `user_prompt` and before the selected context
        of the span (using the `context_getter`).
        It will be formatted specifically for each span, using the `span` variable.
        Example: "Is '{span}' a Colonoscopy (procedure) date?"
    suffix_prompt: Optional[str]
        A suffix prompt to append at the end of the prompt.
    examples : Optional[List[Tuple[str, str]]]
        A list of examples to use for the prompt. Each example is a tuple of
        (input, output). The input is the text to classify and the output is the
        expected classification.
        If None, no examples will be used.
        Example: [("This is a colonoscopy date.", "colonoscopy_date")]
    api_url : str
        The base URL of the vLLM OpenAI-compatible server to call.
        Default: "http://localhost:8000/v1"
    model : str
        The name of the model to use for classification.
        Default: "Qwen/Qwen3-8B"
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
    extra_body : Optional[Dict[str, Any]]
        Additional body parameters to pass to the vLLM API.
        This can be used to pass additional parameters to the model, such as
        `reasoning_parser` or `enable_reasoning`.
    response_format : Optional[Dict[str, Any]]
        The response format to use for the vLLM API call.
        This can be used to specify how the response should be formatted.
    temperature : float
        The temperature for the vLLM API call. Default is 0.0 (deterministic).
    max_tokens : int
        The maximum number of tokens to generate in the response.
        Default is 50.
    response_mapping : Optional[Dict[str, Any]]
        A mapping from regex patterns to values that will be used to map the
        responses from the model to the bindings. If not provided, the raw
        responses will be used. The first matching regex will be used to map the
        response to the binding.
        Example: `{"^yes$": True, "^no$": False}` will map "yes" to True and "no" to
        False.
    timeout : float
        The timeout for the vLLM API call. Default is 15.0 seconds.
    n_concurrent_tasks : int
        The number of concurrent tasks to run when calling the vLLM API.
        Default is 4.
    kwargs: Dict[str, Any]
        Additional keyword arguments passed to the vLLM API call.
        This can include parameters like `n` for the number of responses to generate,
        or any other OpenAI API parameters.

    Authors and citation
    --------------------
    The `eds.llm_qualifier` component was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "span_classifier",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = "Classify the following text:",
        prefix_prompt: Optional[str] = None,
        suffix_prompt: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        api_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B",
        *,
        attributes: AttributesArg = None,
        span_getter: SpanGetterArg = None,
        context_getter: Optional[SpanGetterArg] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 50,
        response_mapping: Optional[Dict[str, Any]] = None,  # TODO change for a function
        timeout: float = 15.0,
        n_concurrent_tasks: int = 4,
        **kwargs,
    ):
        attributes: Attributes
        if attributes is None:
            raise TypeError(
                "The `attributes` parameter is required. Please provide a dict of "
                "attributes to predict or train on."
            )

        span_getter = span_getter or {"ents": True}

        self.bindings: List[Tuple[str, List[str], List[Any]]] = [
            (k if k.startswith("_.") else f"_.{k}", v, [])
            for k, v in attributes.items()
        ]

        # Store API configuration
        self.api_url = api_url
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.prefix_prompt = prefix_prompt
        self.suffix_prompt = suffix_prompt
        self.extra_body = extra_body
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.response_mapping = response_mapping
        self.kwargs = kwargs.get("kwargs") or {}
        self.timeout = timeout
        self.n_concurrent_tasks = n_concurrent_tasks

        self.examples = examples

        super().__init__(nlp, name, span_getter=span_getter)
        self.context_getter = context_getter

        if self.response_mapping:
            self.get_response_mapping_regex_dict()

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
        self.bindings = bindings

    def set_extensions(self):
        super().set_extensions()
        for group in self.bindings:
            qlf = group[0]
            if qlf.startswith("_."):
                qlf = qlf[2:]
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        spans = list(get_spans(doc, self.span_getter))
        spans_text = [span.text for span in spans]
        if self.context_getter is None or not callable(self.context_getter):
            contexts = list(get_spans(doc, self.context_getter))
        else:
            contexts = [self.context_getter(span) for span in spans]

        contexts_text = [context.text for context in contexts]

        doc_batch_messages = []
        for span_text, context_text in zip(spans_text, contexts_text):
            if self.prefix_prompt:
                final_user_prompt = (
                    self.prefix_prompt.format(span=span_text) + context_text
                )
            else:
                final_user_prompt = context_text
            if self.suffix_prompt:
                final_user_prompt += self.suffix_prompt

            messages = create_prompt_messages(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                examples=self.examples,
                final_user_prompt=final_user_prompt,
            )
            doc_batch_messages.append(messages)

        return {
            "$spans": spans,
            "spans_text": spans_text,
            "contexts": contexts,
            "contexts_text": contexts_text,
            "doc_batch_messages": doc_batch_messages,
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> LLMSpanClassifierBatchInput:
        collated = {
            "batch_messages": [
                message for item in batch for message in item["doc_batch_messages"]
            ]
        }

        return collated

    # noinspection SpellCheckingInspection
    def forward(
        self,
        batch: LLMSpanClassifierBatchInput,
    ) -> Dict[str, List[Any]]:
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

        # Here call the LLM API
        llm = AsyncLLM(
            model_name=self.model,
            api_url=self.api_url,
            extra_body=self.extra_body,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format,
            timeout=self.timeout,
            n_concurrent_tasks=self.n_concurrent_tasks,
            **self.kwargs,
        )
        pred = run_async(llm(batch_messages=batch["batch_messages"]))

        return {
            "labels": pred,
        }

    def get_response_mapping_regex_dict(self) -> Dict[str, str]:
        self.response_mapping_regex = {
            re.compile(regex): mapping_value
            for regex, mapping_value in self.response_mapping.items()
        }
        return self.response_mapping_regex

    def map_response(self, value: str) -> str:
        for (
            compiled_regex,
            mapping_value,
        ) in self.response_mapping_regex.items():
            if compiled_regex.search(value):
                mapped_value = mapping_value
                break
            else:
                mapped_value = None
        return mapped_value

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: LLMSpanClassifierBatchOutput,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[Doc]:
        # Preprocessed docs should still be in the cache
        spans = [span for sample in inputs for span in sample["$spans"]]
        all_labels = results["labels"]
        # For each prediction group (exclusive bindings)...

        for qlf, labels, _ in self.bindings:
            for value, span in zip(all_labels, spans):
                if labels is True or span.label_ in labels:
                    if value is None:
                        mapped_value = None
                    elif self.response_mapping is not None:
                        # ...assign the mapped value to the span
                        mapped_value = self.map_response(value)
                    else:
                        mapped_value = value
                    BINDING_SETTERS[qlf](span, mapped_value)

        return docs

    def batch_process(self, docs):
        inputs = [self.preprocess(doc) for doc in docs]
        collated = self.collate(inputs)
        res = self.forward(collated)
        docs = self.postprocess(docs, res, inputs)

        return docs

    def enable_cache(self, cache_id=None):
        # For compatibility
        pass

    def disable_cache(self, cache_id=None):
        # For compatibility
        pass
