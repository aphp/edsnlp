import os
from pathlib import Path
from typing import Optional

import spacy
import typer
from edsnlp.connectors.brat import BratConnector
from spacy.tokens import DocBin
from tqdm import tqdm


def main(
    model: Optional[Path] = typer.Option(None, help="Path to the model"),
    input: Path = typer.Option(
        ..., help="Path to the evaluation dataset, in spaCy format"
    ),
    output: Path = typer.Option(..., help="Path to the output dataset"),
    format: str = typer.Option(..., help="spacy or brat"),
):
    """Partition the data into train/test/dev split."""

    assert format in ("spacy", "brat")

    spacy.require_gpu()

    nlp = spacy.load(model)

    if os.path.isdir(input):
        print("Input format is BRAT")
        input_docs = list(BratConnector(input).brat2docs(nlp))
    else:
        print("Input format is spaCy")
        input_docs = DocBin().from_disk(input).get_docs(nlp.vocab)

    print("Number of docs:", len(input_docs))

    for doc in input_docs:
        doc.ents = []
        doc.spans.clear()

    predicted = []

    nlp.batch_size = 1

    for doc in tqdm(nlp.pipe(input_docs), total=len(input_docs)):
        doc.user_data = {
            k: v
            for k, v in doc.user_data.items()
            if "note_id" in k or "context" in k or "split" in k or "Action" in k or "Allergie" in k or "Certainty" in k or "Temporality" in k or "Family" in k or "Negation" in k
        }
        predicted.append(doc)
    # predicted[0].ents[i]._.negation donne None au lieu de False/True
    if format == "spacy":
        print("Output format is spaCy")
        out_db = DocBin(store_user_data=True, docs=predicted)
        out_db.to_disk(output)
    elif format == "brat":
        print("Output format is BRAT")
        BratConnector(output, attributes=["Negation", "Family", "Temporality", "Certainty", "Action", "Allergie"]).docs2brat(predicted)


if __name__ == "__main__":
    typer.run(main)
