# Qualifying entities

In the previous tutorial, we saw how to match a terminology on a text. Using the `#!python doc.ents` attribute, we can check whether a document mentions a concept of interest to build a cohort or
describe patients.

## The issue

However, consider the classical example where we look for the `diabetes` concept:

=== "French"

    ```
    Le patient n'est pas diabétique.
    Le patient est peut-être diabétique.
    Le père du patient est diabétique.
    ```

=== "English"

    ```
    The patient is not diabetic.
    The patient could be diabetic.
    The patient's father is diabetic.
    ```

None of these expressions should be used to build a cohort: the detected entity is either negated, speculative, or does not concern the patient themself. That's why we need to **qualify the matched
entities**.

!!! warning

    We show an English example just to explain the issue.
    EDS-NLP remains a **French-language** medical NLP library.

## The solution

We can use EDS-NLP's qualifier pipes to achieve that. Let's add specific components to our pipeline to detect these three modalities.

### Adding qualifiers

Adding qualifier pipes is straightforward:

```python hl_lines="25-29"
import edsnlp, edsnlp.pipes as eds

text = (
    "Motif de prise en charge : probable pneumopathie à COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

regex = dict(
    covid=r"(coronavirus|covid[-\s]?19)",
    respiratoire=r"respiratoires?",
)
terms = dict(respiratoire="asthmatique")

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    eds.matcher(
        regex=regex,
        terms=terms,
        attr="LOWER",
    ),
)

nlp.add_pipe(eds.sentences())  # (1)
nlp.add_pipe(eds.negation())  # Negation component
nlp.add_pipe(eds.hypothesis())  # Speculation pipe
nlp.add_pipe(eds.family())  # Family context detection
```

1. Qualifiers pipes need sentence boundaries to be set (see the [specific documentation](../pipes/qualifiers/index.md) for detail).

This code is complete, and should run as is.

### Reading the results

Let's output the results as a pandas DataFrame for better readability:

```python hl_lines="2 34-48"
import edsnlp, edsnlp.pipes as eds
import pandas as pd

text = (
    "Motif de prise en charge : probable pneumopathie à COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

regex = dict(
    covid=r"(coronavirus|covid[-\s]?19)",
    respiratoire=r"respiratoires?",
)
terms = dict(respiratoire="asthmatique")

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    eds.matcher(
        regex=regex,
        terms=terms,
        attr="LOWER",
    ),
)

nlp.add_pipe(eds.sentences())

nlp.add_pipe(eds.negation())  # Negation component
nlp.add_pipe(eds.hypothesis())  # Speculation pipe
nlp.add_pipe(eds.family())  # Family context detection

doc = nlp(text)

# Extraction as a pandas DataFrame
entities = []
for ent in doc.ents:
    d = dict(
        lexical_variant=ent.text,
        label=ent.label_,
        negation=ent._.negation,
        hypothesis=ent._.hypothesis,
        family=ent._.family,
    )
    entities.append(d)

df = pd.DataFrame.from_records(entities)
```

This code is complete, and should run as is.

We get the following result:

| lexical_variant | label        | negation | hypothesis | family |
|:----------------|:-------------|----------|------------|--------|
| COVID19         | covid        | False    | True       | False  |
| respiratoires   | respiratoire | True     | False      | False  |
| asthmatique     | respiratoire | False    | False      | True   |

## Conclusion

The qualifier pipes limits the number of false positives by detecting linguistic modulations such as negations or speculations.
Go to the [full documentation](/pipes/qualifiers/overview) for a complete presentation of the different pipes,
their configuration options and validation performance.
