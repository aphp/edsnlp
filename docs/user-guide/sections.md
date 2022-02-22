# Sections

Detected sections are :

- allergies ;
- antécédents ;
- antécédents familiaux ;
- traitements entrée ;
- conclusion ;
- conclusion entrée ;
- habitus ;
- correspondants ;
- diagnostic ;
- données biométriques entrée ;
- examens ;
- examens complémentaires ;
- facteurs de risques ;
- histoire de la maladie ;
- actes ;
- motif ;
- prescriptions ;
- traitements sortie.

It works by extracting section titles. Then, "sections" cover the entire text that is between two section titles (or the last title and the end of the document).

```{warning}
Use at your own risks : should you rely on `sections` for critical downstream tasks, you should validate the pipeline to make sure that the component works. For instance, the `antecedents` pipeline can use sections to make its predictions, but that possibility is deactivated by default.
```

## Declared extensions

The `eds.sections` pipeline adds two fields to the `doc.spans` attribute :

1. The `section_titles` key contains the list of all section titles extracted using the list declared in the `terms.py` module.
2. The `sections` key contains a list of sections, ie spans of text between two section title (or the last title and the end of the document).

## Usage

The following snippet detects section titles. It is complete and can be run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")

text = "CRU du 10/09/2021\n" "Motif :\n" "Patient admis pour suspicion de COVID"

doc = nlp(text)

doc.spans["section_titles"]
# Out: [Motif :]
```

## Authors and citation

The `eds.sections` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
