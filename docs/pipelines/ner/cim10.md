# CIM10

The `eds.cim10` pipeline component matches the CIM10 (French-language ICD) terminology.

!!! warning "Very low recall"

    When using the `exact' matching mode, this component has a very poor recall performance.
    We can use the `simstring` mode to retrieve approximate matches, albeit at the cost of a significantly higher computation time.

## Usage

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.cim10", config=dict(term_matcher="simstring"))

text = "Le patient est suivi pour fièvres typhoïde et paratyphoïde."

doc = nlp(text)

doc.ents
# Out: (fièvres typhoïde et paratyphoïde,)

ent = doc.ents[0]

ent.label_
# Out: cim10

ent.kb_id_
# Out: A01
```

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.ner.cim10.factory.create_component
    options:
        only_parameters: true

## Authors and citation

The `eds.cim10` pipeline was developed by AP-HP's Data Science team.
