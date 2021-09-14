# Matching a terminology

Matching a terminology is perhaps the most basic application of an NLP pipeline.

EDS-NLP provides a `GenericMatcher` class that simplifies that task, exposing a `matcher` pipeline
that can match on terms, regular expressions, and even fuzzy matching.


## A simple use case : finding COVID

Let us try to find mentions of COVID within a medical note, as well as references to the patient.

```python
import spacy
from edsnlp import components

text = (
    "Motif de prise en charge : pneumopathie à coronavirus\n"
    "Patient atteint de covid 19."
)

terms = dict(covid=["coronavirus", "covid19"], patient="patient")

nlp = spacy.blank("fr")
nlp.add_pipe("matcher", config=dict(terms=terms))

doc = nlp(text)

doc.ents
# Out: (coronavirus,)
```

Let us unpack what happened:

1. We defined a dictionary of terms to look for, in the form `{'label': list of terms}`.
2. We declare a Spacy pipeline, and add the `matcher` component.
3. We apply the pipeline to the texts...
4. And explore the extracted entities.

This example showcases a limitation of our term dictionary : none of `covid 19` and `Patient` where detected by
the pipeline.


## Matching on normalised text

Let us redefine the pipeline :

```python
terms = dict(covid=["coronavirus", "covid19"], patient="patient")

nlp = spacy.blank("fr")
nlp.add_pipe("matcher", config=dict(terms=terms, attr="LOWER"))
```

This time, the pipeline will become case insensitive. Hence :

```python
doc = nlp(text)

doc.ents
# Out: (coronavirus, Patient)
```

We have matched `Patient` ! `covid 19`, however, is still at large. We could write out every
possibility for COVID (eg `covid-19`, `covid 19`, etc), but this might quickly become tedious.


## Using regular expressions

Let us redefine the pipeline once again :

```python
regex = dict(covid="(?i)(coronavirus|covid\s?-?19)", patient="patients?")

nlp = spacy.blank("fr")
nlp.add_pipe("matcher", config=dict(regex=regex))
```

Using regular expressions can help define richer patterns using more compact queries.


## Performing fuzzy matching

The `GenericMatcher` can also perform fuzzy matching ! It is however extremely computationally intensive,
and can easily increase compute times by 60x.
