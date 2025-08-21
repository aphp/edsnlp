# Using a LLM as a span qualifier
In this tutorial we woud learn how to use the `LLMSpanClassifier` pipe to qualify spans.
You should install the extra dependencies before:
```bash
pip install edsnlp[llm]
```

We suppose that there is an available LLM server compatible with OpenAI API.
For example, using the library vllm you can launch an LLM server as follows in command line:
```bash
vllm serve Qwen/Qwen3-8B --port 8000 --enable-prefix-caching --tensor-parallel-size 1 --max-num-seqs=10 --max-num-batched-tokens=35000
```

!!! warning

    As you are probably working with sensitive medical data, Please check whether you can use an external API or if you need to expose an API in your own infrastructure, as in the previous example.

## Import dependencies
```python
from datetime import datetime
from edsnlp.pipes.qualifiers.llm.llm_qualifier import LLMSpanClassifier
from edsnlp.utils.span_getters import make_span_context_getter
import edsnlp, edsnlp.pipes as eds
```
## Define prompt and examples
```python
task_prompts = {
    0: {
        "normalized_task_name": "biopsy_procedure",
        "system_prompt": "You are a medical assistant and you will help answering questions about dates present in clinical notes. Don't answer reasoning. "
        + "We are interested in detecting biopsy dates (either procedure, analysis or result). "
        + "You should answer in a JSON object following this schema {'biopsy':bool}. "
        + "If there is not enough information, answer {'biopsy':'False'}."
        + "\n\n#### Examples:\n",
        "examples": [
            (
                "07/12/2020",
                "07/12/2020 : Anapath / biopsies rectales : Muqueuse rectale normale sous réserve de fragments de petite taille.",
                "{'biopsy':'True'}",
            ),
            (
                "24/12/2021",
                "Chirurgie 24/12/2021 : Colectomie gauche + anastomose colo rectale + clearance hépatique gauche (une méta posée sur",
                "{'biopsy':'False'}",
            ),
        ],
        "prefix_prompt": "\nDetermine if '{span}' corresponds to a biopsy date. The text is as follows:\n<<< ",
        "suffix_prompt": " >>>",
        "json_schema": {
            "properties": {
                "biopsy": {"title": "Biopsy", "type": "boolean"},
            },
            "required": [
                "biopsy",
            ],
            "title": "DateModel",
            "type": "object",
        },
        "response_mapping": {
            "(?i)(oui)|(yes)|(true)": "1",
            "(?i)(non)|(no)|(false)|(don't)|(not)": "0",
        },
    },
}
```

## Format these examples for few-shot learning
```python
def format_examples(raw_examples, prefix_prompt, suffix_prompt):
    examples = []

    for date, context, answer in raw_examples:
        prompt = prefix_prompt.format(span=date) + context + suffix_prompt
        examples.append((prompt, answer))

    return examples
```

## Set parameters and prompts
```python
# Set prompt
prompt_id = 0
raw_examples = task_prompts.get(prompt_id).get("examples")
prefix_prompt = task_prompts.get(prompt_id).get("prefix_prompt")
user_prompt = task_prompts.get(prompt_id).get("user_prompt")
system_prompt = task_prompts.get(prompt_id).get("system_prompt")
suffix_prompt = task_prompts.get(prompt_id).get("suffix_prompt")
examples = format_examples(raw_examples, prefix_prompt, suffix_prompt)

# Define JSON schema
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "DateModel",
        "strict": True,
        "schema": task_prompts.get(prompt_id)["json_schema"],
    },
}

# Set parameters
response_mapping = None
model_name = "Qwen/Qwen3-8B"
api_url = "http://localhost:8000/v1"
max_tokens = 20
extra_body = {
    "chat_template_kwargs": {"enable_thinking": False},
}
temperature = 0
```


## Define the pipeline
```python
nlp = edsnlp.blank("eds")
nlp.add_pipe("sentencizer")
nlp.add_pipe(eds.dates())
nlp.add_pipe(
    LLMSpanClassifier(
        name="llm",
        model=model_name,
        span_getter=["dates"],
        attributes={"_.biopsy_procedure": True},
        context_getter=make_span_context_getter(
            context_sents=(3, 3),
            context_words=(1, 1),
        ),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prefix_prompt=prefix_prompt,
        suffix_prompt=suffix_prompt,
        examples=examples,
        api_url=api_url,
        max_tokens=max_tokens,
        response_mapping=response_mapping,
        extra_body=extra_body,
        temperature=temperature,
        response_format=response_format,
        n_concurrent_tasks=4,
    )
)
```

## Apply it on a document

```python
# Let's try with a fake LLM generated text
text = """
Centre Hospitalier Départemental – RCP Prostate – 20/02/2025

M. Bernard P., 69 ans, retraité, consulte après avoir noté une faiblesse du jet urinaire et des levers nocturnes répétés depuis un an. PSA à 15,2 ng/mL (05/02/2025). TR : nodule ferme sur lobe gauche.

IRM multiparamétrique du 10/02/2025 : lésion PIRADS 5, 2,1 cm, atteinte de la capsule suspectée.
Biopsies du 12/02/2025 : adénocarcinome Gleason 4+4=8, toutes les carottes gauches positives.
Scanner TAP et scintigraphie osseuse du 14/02 : absence de métastases viscérales ou osseuses.

En RCP du 20/02/2025, patient classé cT3a N0 M0, haut risque. Décision : radiothérapie externe + hormonothérapie longue (24 mois). Planification de la simulation scanner le 25/02.
"""
```

```python
t0 = datetime.now()
doc = nlp(text)
t1 = datetime.now()
print("Execution time", t1 - t0)

for span in doc.spans["dates"]:
    print(span, span._.biopsy_procedure)
```

Lets check the type
```python
type(span._.biopsy_procedure)
```
# Apply on multiple documents
```python
docs = [
    doc,
] * 10
predicted_docs = docs.map_pipeline(nlp, 4)
```

```python
t0 = datetime.now()
note_nlp = edsnlp.data.to_pandas(
    predicted_docs,
    converter="ents",
    span_getter="dates",
    span_attributes=[
        "biopsy_procedure",
    ],
)
t1 = datetime.now()
print("Execution time", t1 - t0)
pred.head()
```
