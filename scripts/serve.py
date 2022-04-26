from typing import List

import spacy
from fastapi import FastAPI
from pydantic import BaseModel

import edsnlp

app = FastAPI(title="EDS-NLP", version=edsnlp.__version__)

nlp = spacy.blank("eds")

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
nlp.add_pipe("eds.matcher", config=config)

nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.family")
nlp.add_pipe("eds.hypothesis")
nlp.add_pipe("eds.reported_speech")


class Entity(BaseModel):  # (2)

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


class Document(BaseModel):  # (1)
    text: str
    ents: List[Entity]


@app.post("/process", response_model=List[Document])  # (1)
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
