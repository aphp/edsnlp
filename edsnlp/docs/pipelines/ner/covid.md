# COVID

The `eds.covid` pipeline component detects mentions of COVID19 and adds them to `doc.ents`.

## Usage

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.covid")

text = "Le patient est admis pour une infection au coronavirus."

doc = nlp(text)

doc.ents
# Out: (infection au coronavirus,)
```

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.ner.covid.factory.create_component
    options:
        only_parameters: true

## Authors and citation

The `eds.covid` pipeline was developed by AP-HP's Data Science team.
