# Matching a terminology

Matching a terminology is perhaps the most basic application of a medical NLP pipeline.

In this tutorial, we will cover :

- Matching a terminology using spaCy's matchers, as well as RegExps
- Matching on a specific attribute

You should consider reading the [matcher's specific documentation](../pipes/core/matcher.md) for a description.

!!! note "Comparison to spaCy's matcher"

    spaCy's `Matcher` and `PhraseMatcher` use a very efficient algorithm that compare a hashed representation token by token. **They are not components** by themselves, but can underpin rule-based pipes.

    EDS-NLP's [`RegexMatcher`][edsnlp.matchers.regex.RegexMatcher] lets the user match entire expressions using regular expressions. To achieve this, the matcher has to get to the text representation, match on it, and get back to spaCy's abstraction.

    The `EDSPhraseMatcher` lets EDS-NLP reuse spaCy's efficient algorithm, while adding the ability to skip pollution tokens (see the [normalisation documentation](../pipes/core/normalisation.md) for detail)

## A simple use case : finding COVID19

Let's try to find mentions of COVID19 and references to patients within a clinical note.

```python
import edsnlp, edsnlp.pipes as eds

text = (
    "Motif de prise en charge : probable pneumopathie a COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

terms = dict(
    covid=["coronavirus", "covid19"],
    respiratoire=["asthmatique", "respiratoire"],
)

nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.matcher(terms=terms))

doc = nlp(text)

doc.ents
# Out: (asthmatique,)
```

Let's unpack what happened:

1. We defined a dictionary of terms to look for, in the form `{'label': list of terms}`.
2. We declared a spaCy pipeline, and add the `eds.matcher` component.
3. We applied the pipeline to the texts...
4. ... and explored the extracted entities.

This example showcases a limitation of our term dictionary : the phrases `COVID19` and `difficultés respiratoires` were not detected by the pipeline.

To increase recall, we _could_ just add every possible variation :

```diff
terms = dict(
-    covid=["coronavirus", "covid19"],
+    covid=["coronavirus", "covid19", "COVID19"],
-    respiratoire=["asthmatique", "respiratoire"],
+    respiratoire=["asthmatique", "respiratoire", "respiratoires"],
)
```

But what if we come across `Coronavirus`? Surely we can do better!

## Matching on normalised text

We can modify the matcher's configuration to match on other attributes instead of the verbatim input. You can refer to spaCy's [list of available token attributes](https://spacy.io/usage/rule-based-matching#adding-patterns-attributes){ target=\_blank}.

Let's focus on two:

1. The `LOWER` attribute, which lets you match on a lowercased version of the text.
2. The `NORM` attribute, which adds some basic normalisation (eg `œ` to `oe`). EDS-NLP provides a `eds.normalizer` component that extends the level of cleaning on the `NORM` attribute.

### The `LOWER` attribute

Matching on the lowercased version is extremely easy:

```python
import edsnlp, edsnlp.pipes as eds

text = (
    "Motif de prise en charge : probable pneumopathie a COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

terms = dict(
    covid=["coronavirus", "covid19"],
    respiratoire=["asthmatique", "respiratoire", "respiratoires"],
)

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    eds.matcher(
        terms=terms,
        attr="LOWER",  # (1)
    ),
)

doc = nlp(text)

doc.ents
# Out: (COVID19, respiratoires, asthmatique)
```

1. The matcher's `attr` parameter defines the attribute that the matcher will use. It is set to `"TEXT"` by default (ie verbatim text).

This code is complete, and should run as is.

### Using the normalisation component

EDS-NLP provides its own normalisation component, which modifies the `NORM` attribute in place.
It handles:

- removal of accentuated characters;
- normalisation of quotes and apostrophes;
- lowercasing, which enabled by default in spaCy – EDS-NLP lets you disable it;
- removal of pollution.

!!! note "Pollution in clinical texts"

    EDS-NLP is meant to be deployed on clinical reports extracted from
    hospitals information systems. As such, it is often riddled with
    extraction issues or administrative artifacts that "pollute" the
    report.

    As a core principle, EDS-NLP **never modifies the input text**,
    and `#!python nlp(text).text == text` is **always true**.
    However, we can tag some tokens as pollution elements,
    and avoid using them for matching the terminology.

You can activate it like any other component.

```python hl_lines="4 10 17 23 24"
import edsnlp, edsnlp.pipes as eds

text = (
    "Motif de prise en charge : probable pneumopathie a ===== COVID19, "  # (1)
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

terms = dict(
    covid=["coronavirus", "covid19", "pneumopathie à covid19"],  # (2)
    respiratoire=["asthmatique", "respiratoire", "respiratoires"],
)

nlp = edsnlp.blank("eds")

# Add the normalisation component
nlp.add_pipe(eds.normalizer())  # (3)

nlp.add_pipe(
    eds.matcher(
        terms=terms,
        attr="NORM",  # (4)
        ignore_excluded=True,  # (5)
    ),
)

doc = nlp(text)

doc.ents
# Out: (pneumopathie a ===== COVID19, respiratoires, asthmatique)
```

1. We've modified the example to include a simple pollution.
2. We've added `pneumopathie à covid19` to the list of synonyms detected by the pipeline.
   Note that in the synonym we provide, we kept the accentuated `à`, whereas the example
   displays an unaccentuated `a`.
3. The component can be configured. See the [specific documentation](../pipes/core/normalizer.md) for detail.
4. The normalisation lives in the `NORM` attribute
5. We can tell the matcher to ignore excluded tokens (tokens tagged as pollution by the normalisation component).
   This is not an obligation.

Using the normalisation component, you can match on a normalised version of the text,
as well as **skip pollution tokens during the matching process**.

!!! tip "Using term matching with the normalisation"

    If you use the term matcher with the normalisation, bear in mind that the **examples go through the pipeline**.
    That's how the matcher was able to recover `pneumopathie a ===== COVID19` despite the fact that
    we used an accentuated `à` in the terminology.

    The term matcher matches the input text to the provided terminology, using the selected attribute in both cases.
    The `NORM` attribute that corresponds to `à` and `a` is the same: `a`.

### Preliminary conclusion

We have matched all mentions! However, we had to spell out the singular and plural form of `respiratoire`...
And what if we wanted to detect `covid 19`, or `covid-19` ?
Of course, we _could_ write out every imaginable possibility, but this will quickly become tedious.

## Using regular expressions

Let us redefine the pipeline once again, this time using regular expressions:

```python
import edsnlp, edsnlp.pipes as eds

text = (
    "Motif de prise en charge : probable pneumopathie a COVID19, "
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
        regex=regex,  # (1)
        terms=terms,  # (2)
        attr="LOWER",  # (3)
    ),
)

doc = nlp(text)

doc.ents
# Out: (COVID19, respiratoires, asthmatique)
```

1. We can now match using regular expressions.
2. We can mix and match patterns! Here we keep looking for patients using spaCy's term matching.
3. RegExp matching is not limited to the verbatim text! You can choose to use one of spaCy's native attribute, ignore excluded tokens, etc.

This code is complete, and should run as is.

Using regular expressions can help define richer patterns using more compact queries.

## Visualising matched entities

EDS-NLP is part of the spaCy ecosystem, which means we can benefit from spaCy helper functions.
For instance, spaCy's visualiser displacy can let us visualise the matched entities:

```python
# ↑ Omitted code above ↑

from spacy import displacy

colors = {
    "covid": "orange",
    "respiratoire": "steelblue",
}
options = {
    "colors": colors,
}

displacy.render(doc, style="ent", options=options)
```

If you run this within a notebook, you should get:

<div style="padding: 10px 15px; border: solid 2px; border-radius: 10px; border-color: #afc6e0; font-size: 11pt;">
    <div class="entities" style="line-height: 2.25; direction: ltr">Motif de prise en charge : probable pneumopathie a
    <mark class="entity" style="background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        COVID19
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">covid</span>
    </mark>
    , sans difficultés
    <mark class="entity" style="background: steelblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        respiratoires
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">respiratoire</span>
    </mark>
    </br>Le père du patient est
    <mark class="entity" style="background: steelblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        asthmatique
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">respiratoire</span>
    </mark>
    .</div>
</div>
