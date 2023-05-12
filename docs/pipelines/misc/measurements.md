# Measurements

The `eds.measurements` pipeline's role is to detect and normalise numerical measurements within a medical document.
We use simple regular expressions to extract and normalize measurements, and use `Measurement` classes to store them.

!!! warning

    The ``measurements`` pipeline is still in active development and has not been rigorously validated.
    If you come across a measurement expression that goes undetected, please file an issue !

## Scope

The `eds.measurements` pipeline can extract simple (eg `3cm`) measurements.
It can detect elliptic enumerations (eg `32, 33 et 34kg`) of measurements of the same type and split the measurements accordingly.

The normalized value can then be accessed via the `span._.value` attribute and converted on the fly to a desired unit.

The current pipeline annotates the following measurements out of the box:

| Measurement name | Example                |
|------------------|------------------------|
| `eds.size`       | `1m50`, `1.50m`        |
| `eds.weight`     | `12kg`, `1kg300`       |
| `eds.bmi`        | `BMI: 24`, `24 kg.m-2` |
| `eds.volume`     | `2 cac`, `8ml`         |

## Usage

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe(
    "eds.measurements",
    config=dict(
        measurements=["eds.size", "eds.weight", "eds.bmi"],
        extract_ranges=True,
    ),
)

text = """
Le patient est admis hier, fait 1m78 pour 76kg.
Les deux nodules bénins sont larges de 1,2 et 2.4mm.
BMI: 24.

Le nodule fait entre 1 et 1.5 cm
"""

doc = nlp(text)

measurements = doc.spans["measurements"]

measurements
# Out: [1m78, 76kg, 1,2, 2.4mm, 24, entre 1 et 1.5 cm]

measurements[0]
# Out: 1m78

str(measurements[0]._.value)
# Out: '1.78 m'

measurements[0]._.value.cm
# Out: 178.0

measurements[2]
# Out: 1,2

str(measurements[2]._.value)
# Out: '1.2 mm'

str(measurements[2]._.value.mm)
# Out: 1.2

measurements[4]
# Out: 24

str(measurements[4]._.value)
# Out: '24 kg_per_m2'

str(measurements[4]._.value.kg_per_m2)
# Out: 24

str(measurements[5]._.value)
# Out: 1-1.5 cm
```

To extract all sizes in centimeters, and average range measurements, you can use the following snippet:

```python
sizes = [
    sum(item.cm for item in m._.value) / len(m._.value)
    for m in doc.spans["measurements"]
    if m.label_ == "eds.size"
]
print(sizes)
sizes
# Out: [178.0, 0.12, 0.24, 1.25]
```

## Custom measurement

You can declare custom measurements by changing the patterns

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe(
    "eds.measurements",
    config=dict(
        measurements={
            # this name will be used to define the labels of the matched entities
            "my_custom_surface_measurement": {
                # This measurement unit is homogenous to square meters
                "unit": "m2",
                # To handle cases like "surface: 1.8" (implied m2), we can use
                # unitless patterns
                "unitless_patterns": [
                    {
                        "terms": ["surface", "aire"],
                        "ranges": [
                            {
                                "unit": "m2",
                                "min": 0,
                                "max": 9,
                            }
                        ],
                    }
                ],
            },
        }
    ),
)
```

## Declared extensions

The `eds.measurements` pipeline declares a single [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object,
the `value` attribute that is a `Measurement` instance.

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.misc.measurements.factory.create_component
    options:
        only_parameters: true

## Authors and citation

The `eds.measurements` pipeline was developed by AP-HP's Data Science team.
