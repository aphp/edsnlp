# Deploying as an API

In this section, we will see how you can deploy your pipeline as a REST API using the power of [FastAPI](https://fastapi.tiangolo.com/).

## The NLP pipeline

Let's create a simple NLP model, that can:

- match synonyms of COVID19
- check for negation, speculation and reported speech.

You know the drill:

```python title="pipeline.py"
import spacy

nlp = spacy.blank('fr')

nlp.add_pipe("eds.sentences")

config = dict(
    regex=dict(
        covid=[
            "covid",
            r"covid[-\s]?19",
            r"sars[-\s]?cov[-\s]?2",
            r"corona[-\s]?virus",
        ],
    ),
    attr="LOWER",
)
nlp.add_pipe('eds.matcher', config=config)

nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.family")
nlp.add_pipe("eds.hypothesis")
nlp.add_pipe("eds.reported_speech")
```

## Creating the FastAPI app

FastAPI is a incredibly efficient framework, based on Python type hints from the ground up,
with the help of [Pydantic](https://pydantic-docs.helpmanual.io/) (another great library for building modern Python).
We won't go into too much detail about FastAPI in this tutorial.
For further information on how the framework operates, go to its [excellent documentation](https://fastapi.tiangolo.com/)!

We'll need to create two things:

1. A module containing the models for inputs and outputs.
2. The script that defines the application itself.

```python title="models.py"
from typing import List

from pydantic import BaseModel


class Entity(BaseModel):  # (1)

    # OMOP-style attributes
    start: int
    end: int
    label: str
    lexical_variant: str
    normalized_variant: str

    # Qualifiers
    negated: bool
    hypothesis: bool
    family: bool
    reported_speech: bool


class Document(BaseModel):  # (2)
    text: str
    ents: List[Entity]
```

1. The `Entity` model contains attributes that define a matched entity, as well as variables that contain the output of the qualifier components.
2. The `Document` model contains the input text, and a list of detected entities

Having defined the output models and the pipeline, we can move on to creating the application itself:

```python title="app.py"
from typing import List

from fastapi import FastAPI

from pipeline import nlp
from models import Entity, Document


app = FastAPI(title="EDS-NLP", version=edsnlp.__version__)


@app.post("/covid", response_model=List[Document])  # (1)
async def process(
    notes: List[str],  # (2)
):

    documents = []

    for doc in nlp.pipe(notes):
        entities = []

        for ent in doc.ents:
            entity = Entity(
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
                lexical_variant=ent.text,
                normalized_variant=ent._.normalized_variant,
                negated=ent._.negation,
                hypothesis=ent._.hypothesis,
                family=ent._.family,
                reported_speech=ent._.reported_speech,
            )
            entities.append(entity)

        documents.append(
            Document(
                text=doc.text,
                ents=entities,
            )
        )

    return documents
```

1. By telling FastAPI what output format is expected, you get automatic data validation.
2. In FastAPI, input and output schemas are defined through Python type hinting.
   Here, we tell FastAPI to expect a list of strings in the `POST` request body.
   As a bonus, you get data validation for free.

## Running the API

Our simple API is ready to launch! We'll just need to install FastAPI along with a ASGI server to run it. This can be done in one go:

<div class="termy">

```console
$ pip install fastapi[uvicorn]
---> 100%
Successfully installed fastapi
```

</div>

Launching the API is trivial:

<div class="termy">

```console
$ uvicorn app:app --reload
```

</div>

Go to [`localhost:8000/docs`](http://localhost:8000/docs) to admire the automatically generated documentation!

## Using the API

You can try the API directly from the documentation. Otherwise, you may use the `requests` package:

```python
import request

notes = [
    "Le père du patient n'est pas atteint de la covid.",
    "Probable coronavirus.",
]

r = requests.post(
    "http://localhost:8000/covid",
    json=notes,
)

r.json()
```

You should get something like:

```json
[
  {
    "text": "Le père du patient n'est pas atteint de la covid.",
    "ents": [
      {
        "start": 43,
        "end": 48,
        "label": "covid",
        "lexical_variant": "covid",
        "normalized_variant": "covid",
        "negated": true,
        "hypothesis": false,
        "family": true,
        "reported_speech": false
      }
    ]
  },
  {
    "text": "Probable coronavirus.",
    "ents": [
      {
        "start": 9,
        "end": 20,
        "label": "covid",
        "lexical_variant": "coronavirus",
        "normalized_variant": "coronavirus",
        "negated": false,
        "hypothesis": true,
        "family": false,
        "reported_speech": false
      }
    ]
  }
]
```
