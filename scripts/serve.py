from typing import Dict, List, Optional

import spacy
from fastapi import Body, FastAPI
from pydantic import BaseModel
from spacy.language import Language

import edsnlp
from edsnlp import components  # noqa

app = FastAPI(title="EDS-NLP", version=edsnlp.__version__)


class MatcherConfig(BaseModel):
    terms: Optional[Dict[str, List[str]]] = dict(
        covid=["infection au coronavirus", "covid-19"],
    )
    regex: Optional[Dict[str, List[str]]] = None
    attr: str = "NORM"
    ignore_excluded: bool = True


def get_nlp(config: MatcherConfig) -> Language:

    nlp = spacy.blank("fr")

    nlp.add_pipe("sentences")
    nlp.add_pipe("normalizer")

    nlp.add_pipe(
        "matcher",
        config=config.dict(),
    )

    nlp.add_pipe("negation")
    nlp.add_pipe("hypothesis")
    nlp.add_pipe("rspeech")

    return nlp


@app.get("/")
async def hello():
    return {"message": "Hello World"}


class Notes(BaseModel):
    texts: List[str]


class Entity(BaseModel):
    start: int
    end: int
    label: str
    lexical_variant: str
    normalized_variant: str

    negated: bool
    hypothesis: bool
    reported_speech: bool


class Document(BaseModel):
    text: str
    ents: List[Entity]


@app.post("/process", response_model=List[Document])
async def process(
    notes: List[str] = Body(...),
    config: MatcherConfig = MatcherConfig(),
):

    documents = []

    nlp = get_nlp(config)

    for doc in nlp.pipe(notes):
        documents.append(
            Document(
                text=doc.text,
                ents=[
                    Entity(
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                        lexical_variant=ent.text,
                        normalized_variant=ent._.normalized_variant,
                        negated=ent._.negated,
                        hypothesis=ent._.hypothesis,
                        reported_speech=ent._.reported_speech,
                    )
                    for ent in doc.ents
                ],
            )
        )

    return documents
