# Contextual Matcher

Sometimes, once you've extracted some entities, you need to extract information in its surrounding.
Most of the time, you are looking for:

- Information that could be used to **discard the entity**
- Additional information to enrich **the entity**

For instance, you may want to

- Extract a bunch of diseases, but only if they're symptomatic. In this case, you want to discard every extraction that mentions
"asymptomatic" in its immediate vicinity. While such a filtering is doable using Regular Expression, it basically imposes to modify each RegEx at once.

This pipeline allows for a much easier filtering step.

- Extract the stage of a cancer: To do so you need to
    1. Find mentions of cancer
    2. Find mention of a *stage* and extract its value

  We will see how we can easily do this with the ContextualMatcher.

## The configuration file

The whole ContextualMatcher pipeline is basically defined as a list of **pattern dictionaries**.
Let us see step by step how to build such a list using the example stated just above.

### a. Finding mentions of cancer

To do this, we can build either a set of `terms` or a set of `regex`. `terms` will be used to search for exact matches in the text. While less flexible,
it is faster than using regex. In our case we could use the following lists (which are of course absolutely not exhaustives):

```python3
terms = [
    "cancer",
    "tumeur",
]

regex = [
    "adeno(carcinom|[\s-]?k)",
    "neoplas",
    "melanom",
]
```

Maybe we want to exclude mentions of benine cancers:

```python3
benine = "benign|benin"
```

### b. Find mention of a *stage* and extract its value

For this we will forge a RegEx with one capturing group (basically a pattern enclosed in parenthesis):

```python3
stage = "stade (I{1,3}V?|[1234])"
```

This will extract stage between 1 and 4

We can add a second regex to try to capture if the cancer is in a metastasis stage or not:

```python3
metastase = "(metasta)"
```

### c. The complete configuration

We can now put everything together:

```python3

cancer = dict(
    source="Cancer solide",
    regex=regex,
    terms=terms,
    regex_attr="NORM",
    exclude=dict(
        regex=benine,
        window=3,
    ),
    assign=[
        dict(
            name="stage",
            regex=stage,
            window=(-10,10),
            expand_entity=False,
        ),
        dict(
            name="metastase",
            regex=metastase,
            window=10,
            expand_entity=True,
        ),
    ]
)
```

Here the configuration consists of a single dictionary. We might want to also include lymphoma in the matcher:

```python
lymphome = dict(
    source="Lymphome",
    regex=["lymphom", "lymphangio"],
    regex_attr="NORM",
    exclude=dict(
        regex=["hodgkin"],  # (1)
        window=3,
    ),
)
```

1. We are excluding "Lymphome de Hodgkin" here

In this case, the configuration can be concatenated in a list:

```python3
patterns = [cancer, lymphome]
```


## Usage

```python
import spacy

nlp = spacy.blank("fr")

nlp.add_pipe("sentences")
nlp.add_pipe("normalizer")

nlp.add_pipe(
    "eds.contextual-matcher",
    name="Cancer",
    config=dict(
        patterns=patterns,
    ),
)
```

This snippet is complete, and should run as is. Let us see what we can get from this pipeline with a few examples


=== "Simple match"

    <!-- no-check -->

    ```python3
    txt = "Le patient a eu un cancer il y a 5 ans"
    doc = nlp(txt)
    ent = doc.ents[0]

    ent.label_
    # Out: Cancer

    ent._.source
    # Out: Cancer solide

    ent.text, ent.start, ent.end
    # Out: ('cancer', 5, 6)
    ```

=== "Exclusion rule"

    Let us check that when a *benine* mention is present, the extraction is excluded:

    <!-- no-check -->

    ```python3
    txt = "Le patient a eu un cancer relativement bénin il y a 5 ans"
    doc = nlp(txt)

    doc.ents
    # Out: ()
    ```

=== "Extracting additional infos"

    <!-- no-check -->

    All informations extracted from the provided `assign` configuration can be found in the `assigned` attribute
    under the form of a dictionary:

    ```python3
    txt = "Le patient a eu un cancer de stade 3."
    doc = nlp(txt)

    doc.ents[0]._.assigned
    # Out: {'stage': '3'}
    ```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                                                                                              | Default              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| `patterns`        | Dictionary or List of dictionaries. See below                                                                            |
| `assign_as_span`  | Wether to store eventual extractions defined via the `assign` key as Spans or as string                                  | False                |
| `attr`            | spaCy attribute to match on (eg `NORM`, `LOWER`)                                                                         | `"TEXT"`             |
| `ignore_excluded` | Whether to skip excluded tokens during matching                                                                          | `False`              |
| `regex_flags`     | RegExp flags to use when matching, filtering and assigning (See [here](https://docs.python.org/3/library/re.html#flags)) | 0 (use default flag) |

However, most of the configuration is provided in the `patterns` key, as a **pattern dictionary** or a **list of pattern dictionaries**

## The pattern dictionnary

### Description

The pattern dictionnary is a nested dictionary with the following keys:

=== "`source`"

    A label describing the pattern

=== "`regex`"

    A single Regex or a list of Regexes

=== "`regex_attr`"

    An attributes to overwrite the given `attr` when matching with Regexes.

=== "`terms`"

    A single term or a list of terms (for exact matches)

=== "`exclude`"

    A dictionary (or list of dictionaries) to define exclusion rules. Exclusion rules are given as Regexes, and if a
    match is found in the surrounding context of an extraction, the extraction is removed. Each dictionnary should have the following keys:

    === "`window`"

        Size of the context to use (in number of words). You can provide the window as:

        - A positive integer, in this case the used context will be taken **after** the extraction
        - A negative integer, in this case the used context will be taken **before** the extraction
        - A tuple of integers `(start, end)`, in this case the used context will be the snippet from `start` tokens before the extraction to `end` tokens after the extraction

    === "`regex`"

        A single Regex or a list of Regexes.

=== "`assign`"

    A dictionary to refine the extraction. Similarily to the `exclude` key, you can provide a dictionary to
    use on the context **before** and **after** the extraction.

    === "`name`"

        A name (string)

    === "`window`"

        Size of the context to use (in number of words). You can provide the window as:

        - A positive integer, in this case the used context will be taken **after** the extraction
        - A negative integer, in this case the used context will be taken **before** the extraction
        - A tuple of integers `(start, end)`, in this case the used context will be the snippet from `start` tokens before the extraction to `end` tokens after the extraction

    === "`regex`"

        A dictionary where keys are labels and values are **Regexes with a single capturing group**

    === "`expand_entity`"

        If set to `True`, the initial entity's span will be expanded to the furthest match from the `regex` dictionary


### A full pattern dictionary example

```python3
dict(
    source="AVC",
    regex=[
        "accidents? vasculaires? cerebr",
    ],
    terms="avc",
    regex_attr="NORM",
    exclude=[
        dict(
            regex=["service"],
            window=3,
        ),
        dict(
            regex=[" a "],
            window=-2,
        ),
    ],
    assign=[
        dict(
            name="neo",
            regex=r"(neonatal)",
            expand_entity=True,
            window=3,
        ),
        dict(
            name="trans",
            regex="(transitoire)",
            expand_entity=True,
            window=3,
        ),
        dict(
            name="hemo",
            regex=r"(hemorragique)",
            expand_entity=True,
            window=3,
        ),
        dict(
            name="risk",
            regex=r"(risque)",
            expand_entity=False,
            window=-3,
        ),
    ]
)
```


## Authors and citation

The `eds.matcher` pipeline was developed by AP-HP's Data Science team.
