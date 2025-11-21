import os
import warnings
from collections import deque
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from spacy.tokens import Doc
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.data.converters import DocToMarkupConverter, MarkupToDocConverter
from edsnlp.pipes.base import BaseNERComponent
from edsnlp.utils.fuzzy_alignment import align
from edsnlp.utils.span_getters import SpanGetterArg, SpanSetterArg, get_spans

from ..async_worker import AsyncRequestWorker

# TODO: find a good API for entity attribute extraction too since xml/md converters
#       support attributes.


@registry.factory.register("eds.llm_markup_extractor")
class LlmMarkupExtractor(BaseNERComponent):
    r'''
    The `eds.llm_markup_extractor` component extracts entities using a
    Large Language Model (LLM) by prompting it to annotate the text with a markup
    format (XML or Markdown). The component can be configured with a set of labels
    to extract, and can be provided with few-shot examples to improve performance.

    In practice, along with a system prompt that describes the allowed labels,
    annotation format and few-shot examples (if any), the component sends documents
    to an LLM like:
    ```
    La patient a une néphropathie diabétique.
    ```

    and expects in return the same text annotated with the entities, for instance:
    ```html
    La patient a une <diag>néphropathie diabétique</diag>.
    ```

    which is then parsed to extract the entities. This approach is close to the one
    of [@naguib_2024] but supports various markup formats and multi label
    prompts. Lookup their paper for more details on the prompting strategies and
    performance.

    !!! warning "Experimental"

        This component is experimental. The API and behavior may change in future
        versions. Make sure to pin your `edsnlp` version if you use it in a project.

    !!! note "Dependencies"

        This component requires several dependencies. Run the following command to
        install them:
        ```bash { data-md-color-scheme="slate" }
        pip install openai bm25s Stemmer
        ```
        We recommend even to add them to your `pyproject.toml` or `requirements.txt`.

    Examples
    --------
    If your data is sensitive, we recommend you to use a self-hosted
    model with an OpenAI-compatible API, such as
    [vLLM](https://github.com/vllm-project/vllm).

    Start a server with the model of your choice:

    ```bash { data-md-color-scheme="slate" }
    python -m vllm.entrypoints.openai.api_server \
       --model mistral-small-24b-instruct-2501 \
       --port 8080 \
       --enable-prefix-caching
    ```

    You can then use the `llm_markup_extractor` component as follows:

    <!-- blacken-docs:off -->

    ```python { .no-check }
    import edsnlp, edsnlp.pipes as eds

    prompt = """
    You are a XML-based extraction assistant.
    For every piece of text the user provides, you will rewrite the full text
    word for word, adding XML tags around the relevant pieces of information.

    You must follow these rules strictly:
    - You must only use the provided tags. Do not invent new tags.
    - You must follow the original text exactly: do not alter it, only add tags.
    - You must always close every tag you open.
    - If a piece of text does not contain any of the information to extract, you must return the text unchanged, without any tags.
    - Be consistent in your answers, similar queries must lead to similar answers, do not try to fix your prior answers.
    - Do not add any comment or explanation, just write the text with tags.

    Example with an <noun_group> tag:
    User query: "This is a sample document."
    Assistant answer: "This is <noun_group>a sample document</noun_group>."

    The tags to use are the following:
    - <diag>: A medical diagnosis
    - <treat>: A medical treatment
    """.strip()

    # EDS-NLP util to create documents from Markdown or XML markup.
    # This has nothing to do with the LLM component itself.
    conv = edsnlp.data.converters.MarkupToDocConverter(preset="xml")
    train_docs = [  # (1)!
        conv("Le patient a une <diag>pneumonie</diag>."),
        conv("On prescrit l'<treat>antibiothérapie</treat>."),
        # ... add more examples if you can
    ]

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(
        eds.llm_markup_extractor(
            # OpenAI-compatible API like the local vLLM server above
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            examples=train_docs,
            # Apply the model to each sentence separately
            context_getter="sents",
            # String or function that returns a list of messages (see below)
            prompt=prompt,
            use_retriever=True,
            # For each request, show the model the closest example
            max_few_shot_examples=1,
            # Up to 5 requests in parallel
            max_concurrent_requests=5,
        )
    )
    doc = nlp("Le patient souffre de tuberculose. On débute une antibiothérapie.")
    print([(ent.text, ent.label_) for ent in doc.ents])
    # Out: [('tuberculose', 'diag'), ('antibiothérapie', 'treat')]
    ```

    1. You could also use [EDS-NLP's data API](/data/)
       ```python
       import edsnlp

       train_docs = edsnlp.data.from_iterable(
           [
               "Le patient a une <diag>pneumonie</diag>.",
               "On prescrit l'<treat>antibiothérapie</treat>.",
           ],
           converter="markup",
           preset="xml",
       )
       ```

    <!-- blacken-docs:on -->

    You can also control the prompt more finely by providing a callable
    instead of a string. For instance, let's put all few-shot examples
    in the system message, and the actual user query in a single user message:

    ```python { .no-check }
    def prompt(doc_text, examples):
        system_content = (
            "You are a XML-based extraction assistant.\n"
            "Here are some examples of what's expected:\n"
        )
        for ex_text, ex_markup in examples:
            system_content += f"- User: {ex_text}\n"
            system_content += f"  Bot answer: {ex_markup}\n"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": doc_text},
        ]
    ```

    Parameters
    ----------
    nlp : PipelineProtocol
        Pipeline object.
    name : str
        Component name.
    api_url : str
        The base URL of the OpenAI-compatible API.
        You must explicitly provide this to avoid leaking requests to
        the public OpenAI API. Should you work with sensitive data,
        consider using a self-hosted model.
    model : str
        The model name to use. Must be available on the API server.
    markup_mode : Literal["xml", "md"]
        The markup format to use when formatting the few-shot examples and
        parsing the model's output. Either "xml" (default) or "md" (Markdown).
        Make sure the prompt template matches the chosen format.
    alignment_threshold : float
        The threshold used to align the model's output with the original text.
    prompt : Union[str, Callable[[str, List[Tuple[str, str]]], List[Dict[str, str]]]]
        The prompt is the main way to control the model's behavior.
        It can be either:

        - A string, which will be used as a system prompt.
          Few-shot examples (if any) will be provided as user/assistant
          messages before the actual user query.
        - A callable that takes two arguments:

            * `doc_text`: the text of the document to process
            and returns a list of messages in the format expected by the
            OpenAI chat completions API.
            * `examples`: a list of few-shot examples, each being a tuple
              of (text, markup annotated text)
    examples : Optional[Iterable[Doc]]
        Few-shot examples to provide to the model. The more the better, but
        the total number of tokens in the prompt must be less than the model's
        context size. If `use_retriever` is set to `True`, the most relevant
        examples will be selected automatically.
    max_few_shot_examples : int
        The maximum number of few-shot examples to provide to the model.
        Default to -1 (all examples).
    use_retriever : Optional[bool]
        Whether to use a retriever to select the most relevant few-shot examples.
        If `None` (default), it will be set to `True` if `max_few_shot_examples` is
        greater than 0 and the number of examples is greater than `max_few_shot_examples`.
        If set to `False`, the first `max_few_shot_examples` will be used.
    context_getter : Optional[SpanGetterArg]
        This parameter controls the contexts given to the model for each request.
        It can be used to split the document into smaller chunks, for instance
        sentences by setting `context_getter="sents"`, or process just a part of the
        document, for instance with `context_getter={"sections": "conclusion"}`.
        If `None` (default), the whole document is processed in a single request.
    span_setter : SpanSetterArg
        On which span group (`doc.spans[...]` or `doc.ents`) to set the extracted
        entities.
    span_getter : Optional[SpanGetterArg]
        From which span group (`doc.spans[...]` or `doc.ents`) to get the spans
        to annotate from few-shot examples. Default to the same as `span_setter`.
    seed : Optional[int]
        Optional seed forwarded to the API.
    max_concurrent_requests : int
        Maximum number of concurrent span requests per document.
    api_kwargs : Dict[str, Any]
        Extra keyword arguments forwarded to `chat.completions.create`.
    on_error : Literal["raise", "warn"]
        Error handling strategy. If `"raise"`, exceptions are raised. If `"warn"`,
        exceptions are logged as warnings and processing continues.

    Authors and citation
    --------------------
    The `eds.llm_markup_extractor` component was developed by AP-HP's Data Science
    team.
    '''  # noqa: E501

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "llm_markup_extractor",
        *,
        api_url: str,
        model: str,
        prompt: Union[
            str, Callable[[str, List[Tuple[str, str]]], List[Dict[str, str]]]
        ],
        markup_mode: Literal["xml", "md"] = "xml",
        alignment_threshold: float = 0.0,
        examples: Iterable[Doc] = (),
        max_few_shot_examples: int = -1,
        use_retriever: Optional[bool] = None,
        context_getter: SpanGetterArg = None,
        span_setter: SpanSetterArg = {"ents": True},
        span_getter: Optional[SpanGetterArg] = None,
        seed: Optional[int] = None,
        max_concurrent_requests: int = 1,
        api_kwargs: Optional[Dict[str, Any]] = None,
        on_error: Literal["raise", "warn"] = "raise",
    ):
        import openai

        self.lang = nlp.lang
        self.api_url = api_url
        self.model = model
        if span_getter is None:
            span_getter = span_setter
        self.context_getter = context_getter
        self.markup_to_doc = MarkupToDocConverter(
            preset=markup_mode,
            span_setter=span_setter,
        )
        self.doc_to_markup = DocToMarkupConverter(
            preset=markup_mode,
            span_getter=span_getter,
        )
        self.examples = [(doc.text, self.doc_to_markup(doc)) for doc in examples]
        self.max_few_shot_examples = max_few_shot_examples
        self.span_setter = span_setter
        # Double check just in case, but confit should have caught this
        assert api_url is not None, "api_url must be provided"
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(base_url=self.api_url, api_key=api_key)
        self.async_client = openai.AsyncOpenAI(base_url=self.api_url, api_key=api_key)
        self.prompt = prompt
        self.api_kwargs = api_kwargs or {}
        self.max_concurrent_requests = max_concurrent_requests
        self.on_error = on_error
        self.alignment_threshold = alignment_threshold
        if seed is not None:
            api_kwargs["seed"] = seed
        self.retriever = None
        if self.max_few_shot_examples > 0 and use_retriever is not False:
            self.build_few_shot_retriever_(self.examples)
        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

    def _handle_err(self, msg):
        if self.on_error == "raise":
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    def set_extensions(self) -> None:
        super().set_extensions()

    def __call__(self, doc):
        return self.process(doc)

    def apply_markup_to_doc_(self, doclike: Any, markup_answer: str):
        """
        Apply the markup answer to the doclike object.
        """
        res_text, res_ents = self.markup_to_doc._parse(markup_answer)
        res_text = res_text.rstrip()
        stripped_text = doclike.text.rstrip()

        if stripped_text != res_text:
            ents = [
                {"fragments": [{"begin": s, "end": e}], "label": lab, "attributes": a}
                for s, e, lab, a in res_ents
            ]
            aligned = align(
                {"text": res_text, "entities": ents},
                {"text": stripped_text, "entities": []},
                threshold=self.alignment_threshold,
            )
            res_ents = [
                (f["begin"], f["end"], e["label"], e["attributes"])
                for e in aligned["doc"]["entities"]
                for f in e["fragments"]
            ]

        spans = []
        for start, end, label, attrs in res_ents:
            span = doclike.char_span(start, end, label=label, alignment_mode="expand")
            if span is None:
                continue
            spans.append(span)

        doc = doclike.doc if hasattr(doclike, "doc") else doclike
        self.set_spans(doc, spans)

    def build_few_shot_retriever_(self, samples):
        # TODO: put in a new edsnlp retrievers module?
        import bm25s
        import Stemmer

        lang = {"eds": "french"}.get(self.lang, self.lang)
        stemmer = Stemmer.Stemmer(lang)
        texts = [s[0] for s in samples]
        corpus = bm25s.tokenize(texts, stemmer=stemmer, stopwords=lang)
        retriever = bm25s.BM25()
        retriever.index(corpus)
        self.retriever = retriever
        self.retriever.stemmer = stemmer

    def build_prompt(self, doc):
        import bm25s

        few_shot_examples = []
        if self.retriever is not None:
            closest_texts_indices, scores = self.retriever.retrieve(
                bm25s.tokenize(
                    doc.text,
                    stemmer=self.retriever.stemmer,
                    show_progress=False,
                ),
                k=self.max_few_shot_examples,
                show_progress=False,
            )
            for i in closest_texts_indices[0][: self.max_few_shot_examples]:
                few_shot_examples.append(self.examples[i])
            few_shot_examples = few_shot_examples[::-1]  # reverse to have closest last
        else:
            few_shot_examples = self.examples[: self.max_few_shot_examples]

        if isinstance(self.prompt, str):
            messages = [{"role": "system", "content": self.prompt}]
            for ex_text, ex_markup in few_shot_examples:
                messages.append({"role": "user", "content": ex_text})
                messages.append({"role": "assistant", "content": ex_markup})
            messages.append({"role": "user", "content": doc.text})
        else:
            messages = self.prompt(doc.text, few_shot_examples)
        return messages

    def process(self, doc):
        """
        Handle a single doc
        """
        for ctx in get_spans(doc, self.context_getter):
            messages = self.build_prompt(ctx)
            markup_answer = self._llm_request_sync(messages)
            self.apply_markup_to_doc_(ctx, markup_answer)

        return doc

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """
        Extract entities concurrently, but yield results in the same order
        as the input `docs`. Up to `max_concurrent_requests` span-level
        requests are processed in parallel.

        Parameters
        ----------
        docs: Iterable[Doc]
            Documents to process

        Yields
        ------
        Doc
            Processed documents in the original input order.
        """
        if self.max_concurrent_requests <= 1:  # pragma: no cover
            for doc in docs:
                yield self.process(doc)
            return

        worker = AsyncRequestWorker.instance()

        # Documents that are currently being processed, keyed by their
        # index in the input stream.
        pending_docs: Dict[int, Doc] = {}
        # Number of remaining contexts to process for each document.
        remaining_ctx_counts: Dict[int, int] = {}
        # Fully processed documents waiting to be yielded in order.
        buffer: Dict[int, Doc] = {}
        next_to_yield = 0

        # In-flight LLM requests: task_id -> (doc_index, context)
        in_flight: Dict[int, Tuple[int, Any]] = {}

        docs_iter = enumerate(docs)
        ctx_queue: "deque[Tuple[int, Any]]" = deque()

        def enqueue_new_docs() -> None:
            # Fill the context queue up to `max_concurrent_requests`
            nonlocal docs_iter
            while len(ctx_queue) < self.max_concurrent_requests:
                try:
                    doc_idx, doc = next(docs_iter)
                except StopIteration:
                    break

                pending_docs[doc_idx] = doc
                contexts = list(get_spans(doc, self.context_getter))

                if not contexts:
                    remaining_ctx_counts[doc_idx] = 0
                    buffer[doc_idx] = doc
                else:
                    remaining_ctx_counts[doc_idx] = len(contexts)
                    for ctx in contexts:
                        ctx_queue.append((doc_idx, ctx))

        def submit_until_full() -> None:
            while len(in_flight) < self.max_concurrent_requests and ctx_queue:
                doc_idx, ctx = ctx_queue.popleft()
                messages = self.build_prompt(ctx)
                task_id = worker.submit(self._llm_request_coro(messages))
                in_flight[task_id] = (doc_idx, ctx)

        enqueue_new_docs()
        submit_until_full()

        while in_flight:
            done_task_id = worker.wait_for_any(in_flight.keys())
            result = worker.pop_result(done_task_id)
            doc_idx, ctx = in_flight.pop(done_task_id)

            if result is None:
                pass
            else:
                res, err = result
                if err is not None:
                    self._handle_err(
                        f"[llm_markup_extractor] failed for doc #{doc_idx}: {err!r}"
                    )
                else:
                    try:
                        self.apply_markup_to_doc_(ctx, str(res))
                    except Exception as e:  # pragma: no cover
                        import traceback

                        traceback.print_exc()
                        self._handle_err(
                            f"[llm_markup_extractor] failed to parse result for doc "
                            f"#{doc_idx}: {e!r} in {res!r}"
                        )

            remaining_ctx_counts[doc_idx] -= 1
            if remaining_ctx_counts[doc_idx] == 0:
                buffer[doc_idx] = pending_docs.pop(doc_idx)

            enqueue_new_docs()
            submit_until_full()

            while next_to_yield in buffer:
                yield buffer.pop(next_to_yield)
                next_to_yield += 1

        while next_to_yield in buffer:  # pragma: no cover
            yield buffer.pop(next_to_yield)
            next_to_yield += 1

    def _llm_request_sync(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.api_kwargs,
        )
        return str(response.choices[0].message.content)

    def _llm_request_coro(self, messages) -> Coroutine[Any, Any, str]:
        async def _coro():
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.api_kwargs,
            )
            return response.choices[0].message.content

        return _coro()
