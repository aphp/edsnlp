# Measures

The `eds.measures` pipeline's role is to detect and normalise numerical measures within a medical document.
We use simple regular expressions to extract and normalize measures, and use `Measure` classes to store them.

!!! warning

    The ``measures`` pipeline is still in active development and has not been rigorously validated.
    If you come across a measure expression that goes undetected, please file an issue !

## Scope

The `eds.measures` pipeline can extract simple (eg `3cm`) and composite (eg `3 x 2cm`) measures.
It can detect elliptic enumerations (eg `32, 33 et 34kg`) of measures of the same type and split the measures accordingly.

The normalized value can then be accessed via the `span._.value` attribute and converted on the fly to a desired unit.

The current pipeline supports the following measures:

| Measure name         | Example           | Units                       | Allows composite measures |
| -------------------- | ----------------- | --------------------------- | ------------------------- |
| `eds.measures.size`   | `12m50`, `12.50m` | `mm`, `cm`, `dm`, `m`       | YES                       |
| `eds.measures.weight` | `12kg`, `1g300`   | `mg`, `cg`, `dg`, `g`, `kg` | NO                        |
| `eds.measures.angle`  | `8h`, `8h30`      | `h`                         | NO                        |


## Usage

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe(
    "eds.measures", config=dict(measures=["eds.measures.size", "eds.measures.weight"])
)

text = (
    "Le patient est admis hier, fait 1m78 pour 76kg. "
    "Les deux nodules bénins sont larges de 1,2 et 2.4mm."
    "La tumeur fait 1 x 2cm."
)

doc = nlp(text)

measures = doc.spans["measures"]
# Out: [1m78, 76kg, 1,2, 2.4mm, 1 x 2cm]

measures[0]
# Out: "1m78"

str(measures[0]._.value)
# Out: "1.78m"

measures[0]._.value.cm
# Out: 178.0

measures[-3]
# Out: "1,2"

str(measures[-3]._.value)
# Out: "1.2mm"

str(measures[-3]._.value.mm)
# Out: 1.2

str(measures[-1])
# Out: 1 x 2cm

str(measures[-1]._.value)
# Out: 1.0cm x 2.0cm

str(measures[-1]._.value.mm)
# Out: (10.0, 20.0)

# All measures, simple or composite, support iterable operations
str(max(measures[-1]._.value))
# Out: 2.0cm

str(max(measures[0]._.value))
# Out: 1.78m
```

## Custom measure

You can declare a custom measure using the following code

```python
import spacy
from edsnlp.pipelines.misc.measures import (
    CompositeMeasure,
    SimpleMeasure,
    make_multi_getter,
    make_simple_getter,
)


class CustomCompositeSize(CompositeMeasure):
    mm = property(make_multi_getter("mm"))
    cm = property(make_multi_getter("cm"))
    dm = property(make_multi_getter("dm"))
    m = property(make_multi_getter("m"))


@spacy.registry.misc("eds.measures.custom_size")
class CustomSize(SimpleMeasure):
    # Leave to None if your custom size cannot be composed with others
    COMPOSITE = CustomCompositeSize

    # Optional integer number regex
    INTEGER = r"(?:[0-9]+)"

    # Optional conjunction regex to detect enumeration like "1, 2 et 3cm"
    CONJUNCTIONS = "et|ou"

    # Optional composer regex to detect composite measures like "1 par 2cm"
    COMPOSERS = r"[x*]|par"

    UNITS = {
        # Map of the recognized units
        "cm": {
            # Regex to match prefixes of non abbreviated lexical forms
            "prefix": "centim",
            # Regex to match abbreviated lexical forms
            "abbr": "cm",
            # Scaling value to apply when converting values
            # Only the ratio between different units matter
            "value": 10,
        },
        "mm": {
            "prefix": "mill?im",
            "abbr": "mm",
            "value": 1,
        },
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        """
        Class method to create an instance from the match groups

        int_part: str
            The integer part of the match (eg 12 in 12 metres 50 or 12.50metres)
        dec_part: str
            The decimal part of the match (eg 50 in 12 metres 50 or 12.50metres)
        unit: str
            The normalized variant of the unit (eg "m" for 12 metre 50)
        infix: bool
            Whether the unit was in the before (True) or after (False) the decimal part
        """
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    # Properties to access the numerical value of the measure
    cm = property(make_simple_getter("cm"))
    mm = property(make_simple_getter("mm"))
```

## Declared extensions

The `eds.measures` pipeline declares a single [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object,
the `value` attribute that is a `Measure` instance.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                           |
| ----------------- | ------------------------------------------------ | --------------------------------- |
| `measures`        | The names of the measures to extract, registered in `spacy.misc.registry`          | `["eds.measures.size", "eds.measures.weight", "eds.measures.angle"]` |
| `ignore_excluded` | Whether to ignore excluded tokens for matching           | `False`   |
| `attr`            | spaCy attribute to match on, eg `NORM` or `TEXT` | `"NORM"`                          |

## Authors and citation

The `eds.measures` pipeline was developed by AP-HP's Data Science team.
