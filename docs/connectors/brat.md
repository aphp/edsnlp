# BRAT Connector

BRAT is currently the only supported annotation editor at EDS. BRAT annotations are in the standoff format. Consider the following document:

```
Le patient est admis pour une pneumopathie au coronavirus.
On lui prescrit du paracétamol.
```

It could be annotated as follows :

```
T1	Patient 4 11	patient
T2	Disease 31 58	pneumopathie au coronavirus
T3	Drug 79 90	paracétamol
```

The point of the BRAT connector is to go from the standoff annotation format to an annotated Spacy document :

```python
import spacy
from nlptools.utils.brat import BratConnector

# Instantiate the connector
brat = BratConnector('path/to/brat')

# Instantiate the spacy pipeline
nlp = spacy.blank('fr')

# Convert all BRAT files to a list of documents
docs = brat.brat2docs(nlp)
doc = docs[0]

doc.ents
# Out: [patient, pneumopathie au coronavirus, paracétamol]

doc.ents[0].label_
# Out: Patient
```

The connector can also go the other way around, enabling pre-annotations and an ersatz of active learning.
