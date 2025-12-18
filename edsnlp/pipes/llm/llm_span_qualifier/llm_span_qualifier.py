import json
import os
import warnings
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel
from spacy.tokens import Doc, Span
from typing_extensions import Annotated, Literal

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseSpanAttributeClassifierComponent
from edsnlp.utils.bindings import BINDING_GETTERS, BINDING_SETTERS, AttributesArg
from edsnlp.utils.span_getters import ContextWindow, SpanGetterArg, get_spans

from ..async_worker import AsyncRequestWorker


class LlmSpanQualifier(BaseSpanAttributeClassifierComponent):
    r'''
    The `eds.llm_span_qualifier` component qualifies spans using a
    Large Language Model (LLM) that returns structured JSON attributes.

    This component takes existing spans, wraps them with `<ent>` markers inside a
    context window and prompts an LLM to answer with a JSON object that matches the
    configured schema. The response is validated and written back on the span
    extensions.

    In practice, along with a system prompt that constrains the allowed attributes
    and optional few-shot examples provided as previous user / assistant messages,
    the component sends snippets such as:
    ```
    Biopsies du <date>12/02/2025</date> : adénocarcinome.
    ```

    and expects a minimal JSON answer, for example:
    ```json
    {"biopsy_procedure": "yes"}
    ```
    which is then parsed and assigned to the span attributes.


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

    You can store your OpenAI API key in the `OPENAI_API_KEY` environment
    variable.
    ```python { .no-check }
    import os
    os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    ```

    Start a server with the model of your choice:

    ```bash { data-md-color-scheme="slate" }
    python -m vllm.entrypoints.openai.api_server \
       --model mistral-small-24b-instruct-2501 \
       --port 8080 \
       --enable-prefix-caching
    ```

    You can then use the `llm_span_qualifier` component as follows:

    <!-- blacken-docs:off -->

    === "Yes/no bool classification"

        ```python { .no-check }
        from typing import Annotated, TypedDict
        from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema
        import edsnlp, edsnlp.pipes as eds

        # Pydantic schema used to validate and parse the LLM response
        # The output will be a boolean field.
        # Example:
        # ent._.biopsy_procedure → False
        class BiopsySchema1(BaseModel):
            biopsy_procedure: bool = Field(
                ..., description="Is the span a biopsy procedure or not"
            )

        # Alternative schema using a TypedDict
        # The output will be a dict with a boolean value instead of a boolean field.
        # Example:
        # ent._.biopsy_procedure → {'biopsy_procedure': False}
        class BiopsySchema2(TypedDict):
            biopsy_procedure: bool

        # Alternative annotated schema with custom (de)serializers.
        # This schema transforms the LLM’s output into a boolean before validation.
        # Any case-insensitive variant of "yes", "y", or "true" is interpreted as True;
        # all other values are treated as False.
        #
        # When serializing to JSON, the boolean is converted back into the strings
        # "yes" (for True) or "no" (for False).
        # The output will be a boolean field.
        # Example:
        # ent._.biopsy_procedure → False
        BiopsySchema3 = Annotated[
            bool,
            BeforeValidator(lambda v: str(v).lower() in {"yes", "y", "true"}),
            PlainSerializer(lambda v: "yes" if v else "no", when_used="json"),
        ]


        PROMPT = """
        You are a span classifier. The user sends text where the target is
        marked with <ent>...</ent>. Answer ONLY with a JSON value: "yes" or
        "no" indicating whether the span is a biopsy date.
        """.strip()

        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())
        nlp.add_pipe(eds.dates(span_setter="ents"))

        # EDS-NLP util to create documents from Markdown or XML markup.
        # This has nothing to do with the LLM component itself. The following
        # will create docs with entities labelled "date", store them in doc.ents,
        # and set their span._.biopsy_procedure attribute.
        examples = list(edsnlp.data.from_iterable(
            [
                "IRM du 10/02/2025. Biopsies du <date biopsy_procedure=true>12/02/2025</date> : adénocarcinome.",
                "Chirurgie le 24/12/2021. Colectomie. Consultation du <date biopsy_procedure=false>26/12/2021</date>.",
            ],
            converter="markup",
            preset="xml",
        ).map(nlp.pipes.sentences))

        doc_to_xml = edsnlp.data.converters.DocToMarkupConverter(preset="xml")
        nlp.add_pipe(
            eds.llm_span_qualifier(
                api_url="http://localhost:8080/v1",
                model="mistral-small-24b-instruct-2501",
                prompt=PROMPT,
                span_getter="ents",
                context_getter="sent",
                context_formatter=doc_to_xml,
                attributes=["biopsy_procedure"],
                output_schema=BiopsySchema1, # or BiopsySchema2 or BiopsySchema3
                examples=examples,
                max_few_shot_examples=2,
                max_concurrent_requests=4,
                seed=0,
            )
        )

        text = """
        RCP Prostate – 20/02/2025
        Biopsies du 12/02/2025 : adénocarcinome Gleason 4+4=8.
        Simulation scanner le 25/02/2025.
        """
        doc = nlp(text)
        for d in doc.ents:
            print(d.text, "→ biopsy_procedure:", d._.biopsy_procedure)
        # Out: 20/02/2025 → biopsy_procedure: False
        # Out: 12/02/2025 → biopsy_procedure: True
        # Out: 25/02/2025 → biopsy_procedure: False
        ```

    === "Multi-attribute classification"

        ```python { .no-check }
        from typing import Annotated, Optional
        import datetime
        from pydantic import BaseModel, Field
        import edsnlp, edsnlp.pipes as eds

        # Pydantic schema used to validate the LLM response, serialize the
        # few-shot example answers constrain the model output.
        class CovidMentionSchema(BaseModel):
            negation: bool = Field(..., description="Is the span negated or not")
            date: Optional[datetime.date] = Field(
                None, description="Date associated with the span, if any"
            )

        PROMPT = """
        You are a span classifier. For every piece of markup-annotated text the
        user provides, you predict the attributes of the annotated spans.
        You must follow these rules strictly:
        - Be consistent, similar queries must lead to similar answers.
        - Do not add any comment or explanation, just provide the answer.
        Example with a negation and a date:
        User: "Le 1er mai 2024, le patient a été testé <ent>covid</ent> négatif"
        Assistant: "{"negation": true, "date": "2024-05-01"}"
        For each span, provide a JSON with a "negation" boolean attribute, set to
        true if the span is negated, false otherwise. If a date is associated with
        the span, provide it as a "date" attribute in ISO format (YYYY-MM-DD).
        """.strip()

        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())
        nlp.add_pipe(eds.covid())

        # EDS-NLP util to create documents from Markdown or XML markup.
        # This has nothing to do with the LLM component itself.
        examples = list(edsnlp.data.from_iterable(
            [
                "<ent negation=false date=2024-05-01>Covid</ent> positif le 1er mai 2024.",
                "Pas de <ent negation=true>covid</ent>",
                # ... add more examples if you can
            ],
            converter="markup", preset="xml",
        ).map(nlp.pipes.sentences))

        doc_to_xml = edsnlp.data.converters.DocToMarkupConverter(preset="xml")
        nlp.add_pipe(
            eds.llm_span_qualifier(
                api_url="https://api.openai.com/v1",
                model="gpt-5-mini",
                prompt=PROMPT,
                span_getter="ents",
                context_getter="words[-10:10]",
                context_formatter=doc_to_xml,
                output_schema=CovidMentionSchema,
                examples=examples,
                max_few_shot_examples=1,
                max_concurrent_requests=4,
                seed=0,
            )
        )
        doc = nlp("Pas d'indication de <ent>covid</ent> le 3 mai 2024.")
        (ent,) = doc.ents
        print(ent.text, "→ negation:", ent._.negation, "date:", ent._.date)
        # Out: covid → negation: True date: 2024-05-03
        ```

    <!-- blacken-docs:on -->

    Advanced usage
    --------
    You can also control the prompt more finely by providing a callable instead of a
    string. For example, to put few-shot examples in the system message and keep the
    span context as the user payload:

    ```python { .no-check }
    # Use this for the `prompt` argument instead of PROMPT above
    def prompt(context_text, examples):
        messages = []
        system_content = (
            "You are a span classifier.\n"
            "Answer with JSON using the keys: biopsy_procedure.\n"
            "Here are some examples:\n"
        )
        for ex_context, ex_json in examples:
            system_content += f"- Context: {ex_context}\n"
            system_content += f"  JSON: {ex_json}\n"
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": context_text})
        return messages
    ```

    You can also control the context formatting by providing a custom callable
    to the `context_formatter` parameter. For example, to wrap the context with
    a custom prefix and suffix as follows:

    ```python { .no-check }
    from spacy.tokens import Doc

    class ContextFormatter:
        def __init__(self, prefix: str, suffix: str):
            self.prefix = prefix
            self.suffix = suffix

        def __call__(self, context: Doc) -> str:
            span = context.ents[0].text if context.ents else ""
            prefix = self.prefix.format(span=span)
            suffix = self.suffix.format(span=span)
            return f"{prefix}{context.text}{suffix}"

    context_formatter = ContextFormatter(prefix="\n## Context\n\n<<<\n",
                                         suffix= "\n>>>\n\n## Instruction\nDoes '{span}' corresponds to a Biopsy date?")
    ```

    !!! note "`max_concurrent_requests` parameter"

        We recommend setting the `max_concurrent_requests` parameter to a greater value
          to improve throughput when processing batches of documents.

    Parameters
    ----------
    nlp : PipelineProtocol
        Pipeline object.
    name : str
        Component name.
    api_url : str
        Base URL of the OpenAI-compatible API.
    model : str
        Model identifier exposed by the API.
    prompt : Union[str, Callable[[Union[str, Doc], List[Tuple[Union[str, Doc], str]]], List[Dict[str, str]]]]
        The prompt is the main way to control the model's behavior.
        It can be either:

        - A string, which will be used as a system prompt.
          Few-shot examples (if any) will be provided as user/assistant
          messages before the actual user query.
        - A callable that takes three arguments and returns a list of messages in the
          format expected by the OpenAI chat completions API.

            * `context`: the context text with the target span marked up
            * `examples`: a list of few-shot examples, each being a tuple of
                (context, answer)
    span_getter : Optional[SpanGetterArg]
        Spans to classify. Defaults to `{"ents": True}`.
    context_getter : Optional[ContextWindow]
        Optional context window specification (e.g. `"sent"`, `"words[-10:10]"`).
        If `None`, the whole document text is used.
    context_formatter : Optional[Callable[[Doc], str]]
        Callable used to render the context passed to the LLM. Defaults to
        `lambda context_getter_output:  context_getter_output.text`.
    attributes : Optional[AttributesArg]
        Attributes to predict. If omitted, the keys are inferred from the provided
        schema.
    output_schema : Optional[Union[Type[BaseModel], Type[Any], Annotated[Any, Any]]]
        Pydantic model class used to validate responses and serialise few-shot
        examples. If the schema is a mapping/object, it will also be used to
        force the model to output a valid JSON object.
    examples : Optional[Iterable[Doc]]
        Few-shot examples used in prompts.
    max_few_shot_examples : int
        Maximum number of few-shot examples per request (`-1` means all).
    use_retriever : Optional[bool]
        Whether to select few-shot examples with BM25 (defaults to automatic choice).
        If there are few shot examples and `max_few_shot_examples > 0`, this enabled
        by default.
    seed : Optional[int]
        Optional seed forwarded to the API.
    max_concurrent_requests : int
        Maximum number of concurrent span requests per batch of documents.
    api_kwargs : Dict[str, Any]
        Extra keyword arguments forwarded to `chat.completions.create`.
    on_error : Literal["raise", "warn"]
        Error handling strategy. If `"raise"`, exceptions are raised. If `"warn"`,
        exceptions are logged as warnings and processing continues.
    timeout : Optional[float]
        Optional timeout (in seconds) for each LLM request.
    default_headers : Optional[Dict[str, str]]
        Optional default headers for the API client.
    '''  # noqa: E501

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "llm_span_qualifier",
        *,
        api_url: str,
        model: str,
        prompt: Union[
            str,
            Callable[
                [Union[str, Doc], List[Tuple[Union[str, Doc], str]]],
                List[Dict[str, str]],
            ],
        ],
        span_getter: Optional[SpanGetterArg] = None,
        context_getter: Optional[ContextWindow] = None,
        context_formatter: Optional[Callable[[Doc], Union[str, Doc]]] = None,
        attributes: Optional[AttributesArg] = None,  # confit will auto cast to dict
        output_schema: Optional[
            Union[
                Type[BaseModel],
                Type[Any],
                Annotated[Any, Any],
            ]
        ] = None,
        examples: Optional[Iterable[Doc]] = None,
        max_few_shot_examples: int = -1,
        use_retriever: Optional[bool] = None,
        seed: Optional[int] = None,
        max_concurrent_requests: int = 1,
        api_kwargs: Optional[Dict[str, Any]] = None,
        on_error: Literal["raise", "warn"] = "raise",
        timeout: Optional[float] = None,
        default_headers: Optional[Dict[str, str]] = {"Connection": "close"},
    ):
        import openai

        span_getter = span_getter or {"ents": True}
        self.lang = nlp.lang
        self.api_url = api_url
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.context_window = (
            ContextWindow.validate(context_getter)
            if context_getter is not None
            else None
        )
        self.context_formatter = context_formatter or (
            lambda context_getter_output: context_getter_output.text
        )
        self.seed = seed
        self.api_kwargs = api_kwargs or {}
        self.max_concurrent_requests = max_concurrent_requests
        self.on_error = on_error

        if attributes is None:
            if hasattr(output_schema, "model_fields"):
                attr_map = {name: True for name in output_schema.model_fields.keys()}
            else:
                raise ValueError(
                    "You must provide either `attributes` or a valid pydantic"
                    "`output_schema` for llm_span_qualifier."
                )
        else:
            attr_map = attributes

        self.scalar_schema = True
        if output_schema is not None:
            try:
                self.scalar_schema = not issubclass(output_schema, BaseModel)
            except TypeError:
                self.scalar_schema = True

        if self.scalar_schema and output_schema is not None:
            if not attributes or len(attributes) != 1:
                raise ValueError(
                    "When the provided output schema is a scalar type, you must "
                    "provide exactly one attribute."
                )

            # This class name is produced in the json output_schema so the model
            # may see this depending on API implementation !
            from pydantic import RootModel

            class Output(RootModel):
                root: output_schema  # type: ignore

            self.output_schema = Output  # type: ignore

        else:
            self.output_schema = output_schema

        self.response_format = (
            self._build_response_format(self.output_schema)
            if self.output_schema
            else None
        )

        self.bindings: List[
            Tuple[str, Union[bool, List[str]], str, Callable, Callable]
        ] = []
        for attr_name, labels in attr_map.items():
            if (
                attr_name.startswith("_.")
                or attr_name.endswith("_")
                or attr_name in {"label_", "kb_id_"}
            ):
                attr_path = attr_name
            else:
                attr_path = f"_.{attr_name}"
            json_key = attr_path[2:] if attr_path.startswith("_.") else attr_path
            setter = BINDING_SETTERS[attr_path]
            getter = BINDING_GETTERS[attr_path]
            self.bindings.append((attr_path, labels, json_key, setter, getter))
        self.attributes = {path: labels for path, labels, *_ in self.bindings}

        self.examples: List[Tuple[Union[str, Doc], str]] = []
        for doc in examples or []:
            for span in get_spans(doc, span_getter):
                context_doc = self._build_context_doc(span)
                formatted_context = self.context_formatter(context_doc)
                values: Dict[str, Any] = {}
                for _, labels, json_key, _, getter in self.bindings:
                    if (
                        labels is False
                        or labels is not True
                        and span.label_ not in (labels or [])
                    ):
                        continue
                    try:
                        values[json_key] = getter(span)
                    except Exception:  # pragma: no cover
                        self._handle_err(
                            f"Failed to get attribute {attr_path!r} for span "
                            f"{span.text!r} in example doc {span.doc._.note_id}"
                        )
                if self.scalar_schema:
                    values = next(iter(values.values()))
                if self.output_schema is not None:
                    try:
                        answer = self.output_schema.model_validate(
                            values
                        ).model_dump_json(exclude_none=True)
                    except Exception:  # pragma: no cover
                        self._handle_err(
                            f"[llm_span_qualifier] Failed to validate example "
                            f"values against the output schema: {values!r}"
                        )
                        continue
                else:
                    answer = json.dumps(values)
                self.examples.append((formatted_context, answer))

        self.max_few_shot_examples = max_few_shot_examples
        self.retriever = None
        self.retriever_stemmer = None
        if self.max_few_shot_examples > 0 and use_retriever is not False:
            self.build_few_shot_retriever_(self.examples)

        api_key = os.getenv(
            "OPENAI_API_KEY", "EMPTY_API_KEY"
        )  # API key should be non empty (even when exposing local models without auth)
        self.client = openai.Client(base_url=self.api_url, api_key=api_key)
        self._async_client = openai.AsyncOpenAI(
            base_url=self.api_url,
            api_key=api_key,
            default_headers=default_headers,
        )

        super().__init__(nlp=nlp, name=name, span_getter=span_getter)

    def _handle_err(self, msg):
        if self.on_error == "raise":
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    def set_extensions(self) -> None:
        super().set_extensions()
        for attr_path, *_ in self.bindings:
            if attr_path.startswith("_."):
                ext_name = attr_path[2:].split(".")[0]
                if not Span.has_extension(ext_name):
                    Span.set_extension(ext_name, default=None)

    def build_few_shot_retriever_(
        self, samples: List[Tuple[Union[str, Doc], str]]
    ) -> None:
        # Same BM25 strategy as llm_markup_extractor
        import bm25s
        import Stemmer

        lang = {"eds": "french"}.get(self.lang, self.lang)
        stemmer = Stemmer.Stemmer(lang)
        corpus = bm25s.tokenize(
            [
                sample.text if isinstance(sample, Doc) else sample
                for sample, _ in samples
            ],
            stemmer=stemmer,
            stopwords=lang,
        )
        retriever = bm25s.BM25()
        retriever.index(corpus)
        self.retriever = retriever
        self.retriever_stemmer = stemmer

    def build_prompt(self, formatted_context: Union[str, Doc]) -> List[Dict[str, str]]:
        """Build the prompt messages for the LLM request."""
        import bm25s

        if isinstance(formatted_context, Doc):
            context_text = formatted_context.text
        else:
            context_text = formatted_context

        few_shot_examples: List[Tuple[str, str]] = []
        if self.retriever is not None:
            closest, _ = self.retriever.retrieve(
                bm25s.tokenize(
                    context_text,
                    stemmer=self.retriever_stemmer,
                    show_progress=False,
                ),
                k=self.max_few_shot_examples,
                show_progress=False,
            )
            for i in closest[0][: self.max_few_shot_examples]:
                few_shot_examples.append(self.examples[i])
            few_shot_examples = few_shot_examples[::-1]
        else:
            few_shot_examples = self.examples[: self.max_few_shot_examples]

        if isinstance(self.prompt, str):
            messages = [{"role": "system", "content": self.prompt}]
            for ctx, ans in few_shot_examples:
                messages.append({"role": "user", "content": ctx})
                messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": context_text})
            return messages
        return self.prompt(formatted_context, few_shot_examples)

    def _llm_request_sync(self, messages: List[Dict[str, str]]) -> str:
        call_kwargs = dict(self.api_kwargs)
        if "response_format" not in call_kwargs and self.response_format is not None:
            call_kwargs["response_format"] = self.response_format
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=self.seed,
            **call_kwargs,
        )
        return response.choices[0].message.content or ""

    def _llm_request_coro(
        self, messages: List[Dict[str, str]]
    ) -> Coroutine[Any, Any, str]:
        async def _coro():
            call_kwargs = dict(self.api_kwargs)
            if (
                "response_format" not in call_kwargs
                and self.response_format is not None
            ):
                call_kwargs["response_format"] = self.response_format
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                seed=self.seed,
                **call_kwargs,
            )
            return response.choices[0].message.content or ""

        return _coro()

    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        data = text
        if self.output_schema is not None:
            if self.scalar_schema:
                start = 0
                end = len(text)
            else:
                if text.startswith("```"):
                    text = text.strip("`")
                    text = text.strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                start = text.find("{")
                end = text.rfind("}") + 1
            if start == -1 or end <= 0 or end <= start:
                return None
            try:
                data = self.output_schema.model_validate_json(text).model_dump()
            except Exception:
                try:
                    # Interpret as a string
                    data = self.output_schema.model_validate_json(
                        json.dumps(text)
                    ).model_dump()
                except Exception:  # pragma: no cover
                    self._handle_err(
                        "[llm_span_qualifier] Failed to validate LLM response"
                        f" against the output schema: {text!r}",
                    )
                    data = raw
        if self.scalar_schema:
            data = {next(iter(self.attributes.keys())): data}
        return data

    def _build_context_doc(self, span: Span) -> Doc:
        ctx_source = (
            span.doc[:] if self.context_window is None else self.context_window(span)
        )
        context = ctx_source.as_doc()
        offset = ctx_source.start
        rel_start = max(0, span.start - offset)
        rel_end = max(rel_start, min(len(context), span.end - offset))
        ent = Span(context, rel_start, rel_end, label=span.label_)
        context.ents = (ent,)
        return context

    def _set_values_on_span(self, span: Span, data: Optional[Dict[str, Any]]) -> None:
        for attr_path, labels, json_key, setter, _ in self.bindings:
            if (
                labels is False
                or labels is not True
                and span.label_ not in (labels or [])
            ):
                continue
            # only when scalar mode we use attr_path as key, maybe change that later ?
            value = (
                None
                if data is None
                else data.get(json_key)
                if json_key in data
                else data.get(attr_path)
            )
            try:
                setter(span, value)
            except Exception as exc:  # pragma: no cover
                self._handle_err(
                    f"[llm_span_qualifier] Failed to set attribute {attr_path!r} "
                    f"for span {span.text!r} in doc {span.doc._.note_id}: {exc!r}"
                )

    def _build_response_format(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        raw_schema = schema.model_json_schema()
        json_schema = json.loads(json.dumps(raw_schema))

        if isinstance(json_schema, dict):
            json_schema.setdefault("type", "object")
            json_schema.setdefault("additionalProperties", False)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__.replace(" ", "_"),
                "schema": json_schema,
            },
        }

    def _process_docs_async(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        worker = AsyncRequestWorker.instance()
        pending: Dict[int, Tuple[Dict[str, Any], Span]] = {}
        doc_states: List[Dict[str, Any]] = []
        docs_iter = iter(docs)
        exhausted = False
        next_yield = 0

        def make_state(doc: Doc) -> Dict[str, Any]:
            spans = list(get_spans(doc, self.span_getter))
            return {
                "doc": doc,
                "spans": spans,
                "next_span": 0,
                "pending": 0,
            }

        def doc_done(state: Dict[str, Any]) -> bool:
            return state["next_span"] >= len(state["spans"]) and state["pending"] == 0

        def schedule() -> None:
            if next_yield >= len(doc_states):
                return
            for state in doc_states[next_yield:]:
                while (
                    state["next_span"] < len(state["spans"])
                    and len(pending) < self.max_concurrent_requests
                ):
                    span = state["spans"][state["next_span"]]
                    state["next_span"] += 1
                    context_doc = self._build_context_doc(span)
                    formatted_context = self.context_formatter(context_doc)
                    messages = self.build_prompt(formatted_context)
                    task_id = worker.submit(
                        self._llm_request_coro(messages), timeout=self.timeout
                    )
                    pending[task_id] = (state, span)
                    state["pending"] += 1
                    if len(pending) >= self.max_concurrent_requests:
                        return

        while True:
            while not exhausted and len(pending) < self.max_concurrent_requests:
                try:
                    doc = next(docs_iter)
                except StopIteration:
                    exhausted = True
                    break
                doc_states.append(make_state(doc))
                schedule()

            while next_yield < len(doc_states) and doc_done(
                doc_states[next_yield]
            ):  # pragma: no cover
                yield doc_states[next_yield]["doc"]
                next_yield += 1

            if exhausted and len(pending) == 0 and next_yield == len(doc_states):
                break

            if len(pending) == 0:  # pragma: no cover
                if exhausted and next_yield == len(doc_states):
                    break
                continue

            done_task = worker.wait_for_any(pending.keys())
            result = worker.pop_result(done_task)
            state, span = pending.pop(done_task)
            state["pending"] -= 1
            raw = None
            err = None
            if result is not None:
                raw, err = result
            if err is not None:  # pragma: no cover
                self._handle_err(
                    f"[llm_span_qualifier] request failed for span "
                    f"'{span.text}' in doc {span.doc._.note_id}: {err!r}"
                )
                data = None
            else:
                data = self._parse_response(str(raw))
                if data is None:  # pragma: no cover
                    self._handle_err(
                        "[llm_span_qualifier] Failed to parse LLM response for span "
                        f"'{span.text}' in doc {span.doc._.note_id}: {raw!r}"
                    )
            self._set_values_on_span(span, data)
            schedule()
            while next_yield < len(doc_states) and doc_done(doc_states[next_yield]):
                yield doc_states[next_yield]["doc"]
                next_yield += 1

    def process(self, doc: Doc) -> Doc:
        spans = list(get_spans(doc, self.span_getter))

        for span in spans:
            context_doc = self._build_context_doc(span)
            formatted_context = self.context_formatter(context_doc)
            messages = self.build_prompt(formatted_context)
            data = None
            try:
                raw = self._llm_request_sync(messages)
            except Exception as err:  # pragma: no cover
                self._handle_err(
                    "[llm_span_qualifier] request failed for span "
                    f"'{span.text}' in doc {doc._.note_id}: {err!r}"
                )
            else:
                data = self._parse_response(raw)
                if data is None:  # pragma: no cover
                    self._handle_err(
                        "[llm_span_qualifier] Failed to parse LLM response for span "
                        f"'{span.text}' in doc {doc._.note_id}: {raw!r}"
                    )
            self._set_values_on_span(span, data)
        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        if self.max_concurrent_requests <= 1:  # pragma: no cover
            for doc in docs:
                yield self(doc)
            return

        yield from self._process_docs_async(docs)
