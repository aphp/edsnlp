# Matching a terminology

Matching a terminology is perhaps the most basic application of a medical NLP pipeline.

In this tutorial, we will cover :

- Matching a terminology using SpaCy's matchers, as well as RegExps
- Matching on a specific attribute

You should consider reading the [matcher's specific documentation](/pipelines/core/matcher.md) for a description.

!!! note "Comparison to SpaCy's matcher"

    SpaCy's `Matcher` and `PhraseMatcher` use a very efficient algorithm that compare a hashed representation token by token. **They are not components** by themselves, but can underpin rule-based pipelines.

    EDS-NLP's [`RegexMatcher`][edsnlp.matchers.regex.RegexMatcher] lets the user match entire expressions using regular expressions. To achieve this, the matcher has to get to the text representation, match on it, and get back to SpaCy's abstraction.

    The [`EDSPhraseMatcher`][edsnlp.matchers.phrase.EDSPhraseMatcher] lets EDS-NLP reuse SpaCy's efficient algorithm, while adding the ability to skip pollution tokens (see the [normalisation documentation](/pipelines/core/normalisation.md) for detail)

## A simple use case : finding COVID

Let's try to find mentions of COVID and references to patients within a medical note.

```python
import spacy

text = (
    "Motif de prise en charge : probable pneumopathie à COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

terms = dict(
    covid=["coronavirus", "covid19"],
    respiratoire=["asthmatique", "respiratoire"],
)

nlp = spacy.blank("fr")
nlp.add_pipe("eds.matcher", config=dict(terms=terms))

doc = nlp(text)

doc.ents
# Out: (asthmatique)
```

Let's unpack what happened:

1. We defined a dictionary of terms to look for, in the form `{'label': list of terms}`.
2. We declared a SpaCy pipeline, and add the `eds.matcher` component.
3. We applied the pipeline to the texts...
4. ... and explored the extracted entities.

This example showcases a limitation of our term dictionary : the phrases `COVID19` and `difficultés respiratoires` where not detected by the pipeline.

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

## Matching on normalized text

We can modify the matcher's configuration to match on lowercase text instead of the verbatim input :

```python
import spacy

text = (
    "Motif de prise en charge : probable pneumopathie à COVID19, "
    "sans difficultés respiratoires\n"
    "Le père du patient est asthmatique."
)

terms = dict(
    covid=["coronavirus", "covid19"],
    respiratoire=["asthmatique", "respiratoire", "respiratoires"],
)

nlp = spacy.blank("fr")
nlp.add_pipe(
    "eds.matcher",
    config=dict(
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

We have matched `COVID19` and `respiratoires` ! However, we had to spell out the singular and plural form of `respiratoire`... And what if we wanted to detect `covid 19`, or `covid-19` ? We could write out every imaginable possibility, but this will quickly become tedious.

## Using regular expressions

Let us redefine the pipeline once again, this time using regular expressions:

```python
import spacy

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

nlp = spacy.blank("fr")
nlp.add_pipe(
    "eds.matcher",
    config=dict(
        regex=regex,  # (1)
        terms=terms,  # (2)
        attr="LOWER",
    ),
)

doc = nlp(text)

doc.ents
# Out: (COVID19, respiratoires, asthmatique)
```

1. We can now match using regular expressions.
2. We can mix and match patterns! Here we keep looking for patients using SpaCy's term matching.

This code is complete, and should run as is.

Using regular expressions can help define richer patterns using more compact queries.
