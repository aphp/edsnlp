
# Contextual Matcher

During feature extraction, it may be necessary to search for additional patterns in their neighborhood, namely:

- patterns to discard irrelevant entities
- patterns to enrich these entities and store some information

For example, to extract mentions of non-benign cancers, we need to discard all extractions that mention "benin" in their immediate neighborhood.
Although such a filtering is feasible using a regular expression, it essentially requires modifying each of the regular expressions.

The ContextualMatcher allows to perform this extraction in a clear and concise way.

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

Maybe we want to exclude mentions of benign cancers:

```python3
benign = "benign|benin"
```

### b. Find mention of a *stage* and extract its value

For this we will forge a RegEx with one capturing group (basically a pattern enclosed in parentheses):

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
        regex=benign,
        window=3,
    ),
    assign=[
        dict(
            name="stage",
            regex=stage,
            window=(-10,10),
            replace_entity=True,
            reduce_mode=None,
        ),
        dict(
            name="metastase",
            regex=metastase,
            window=10,
            replace_entity=False,
            reduce_mode="keep_last",
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

## Available parameters for more flexibility

3 main parameters can be used to refine how entities will be formed

### The `include_assigned` parameter

Following the previous example, you might want your extracted entities to **include**, if found, the cancer stage and the metastasis status. This can be achieved by setting `include_assigned=True` in the pipe configuration.

For instance, from the sentence "Le patient a un cancer au stade 3", the extracted entity will be:

- "cancer" if `include_assigned=False`
- "cancer au stade 3" if `include_assigned=True`

### The `reduce_mode` parameter

It may happen that an assignment matches more than once. For instance, in the (nonsensical) sentence "Le patient a un cancer au stade 3 et au stade 4", both "stade 3" and "stade 4" will be matched by the `stage` assign key. Depending on your use case, you may want to keep all the extractions, or just one.

- If `reduce_mode=None` (default), all extractions are kept in a list
- If `reduce_mode="keep_first"`, only the extraction closest to the main matched entity will be kept (in this case, it would be "stade 3" since it is the closest to "cancer")
- If `reduce_mode=="keep_last"`, only the furthest extraction is kept.

### The `replace_entity` parameter

This parameter can be se to `True` **only for a single assign key per dictionary**. This limitation comes from the purpose of this parameter: If set to `True`, the corresponding `assign` key will be returned as the entity, instead of the match itself. For clarity, let's take the same sentence "Le patient a un cancer au stade 3" as an example:

- if `replace_entity=True` in the `stage` assign key, then the extracted entity will be "stade 3" instead of "cancer"
- if `replace_entity=False` for every assign key, the returned entity will be, as expected, "cancer"

**Please notice** that with `replace_entity` set to True, if the correponding assign key matches nothing, the entity will be discarded.


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

Let us see what we can get from this pipeline with a few examples


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

    Let us check that when a *benign* mention is present, the extraction is excluded:

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

::: edsnlp.pipelines.core.contextual_matcher.factory.create_component
    options:
        only_parameters: true

However, most of the configuration is provided in the `patterns` key, as a **pattern dictionary** or a **list of pattern dictionaries**

## The pattern dictionary

### Description

A patterr is a nested dictionary with the following keys:

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
    match is found in the surrounding context of an extraction, the extraction is removed. Each dictionary should have the following keys:

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

    === "`replace_entity`"

        If set to `True`, the match from the corresponding assign key will be used as entity, instead of the main match. See [this paragraph][the-replace_entity-parameter]

    === "`reduce_mode`"

        Set how multiple assign matches are handled. See the documentation of the [`reduce_mode` parameter][the-reduce_mode-parameter]

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
