# spaCy 101

EDS-NLP is a spaCy library. To use it, you will need to familiarise yourself with some key spaCy concepts.

!!! tip "Skip if you're familiar with spaCy"

    This page is intended as a crash course for the very basic spaCy concepts that are needed to use EDS-NLP.
    If you've already used spaCy, you should probably skip to the next page.

In a nutshell, spaCy offers three things:

- a convenient abstraction with a language-dependent, rule-based, deterministic and non-destructive tokenizer
- a rich set of rule-based and trainable components
- a configuration and training system

We will focus on the first item.

Be sure to check out [spaCy's crash course page](https://spacy.io/usage/spacy-101) for more information on the possibilities offered by the library.

## Resources

The [spaCy documentation](https://spacy.io/) is one of the great strengths of the library.
In particular, you should check out the ["Advanced NLP with spaCy" course](https://course.spacy.io/en/),
which provides a more in-depth presentation.

## spaCy in action

Consider the following minimal example:

```python
import spacy  # (1)

# Initialise a spaCy pipeline
nlp = spacy.blank("fr")  # (2)

text = "Michel est un penseur latéral."  # (3)

# Apply the pipeline
doc = nlp(text)  # (4)

doc.text
# Out: 'Michel est un penseur latéral.'
```

1.  Import spaCy...
2.  Load a pipeline. In spaCy, the `nlp` object handles the entire processing.
3.  Define a text you want to process.
4.  Apply the pipeline and get a spaCy [`Doc`](https://spacy.io/api/doc) object.

We just created a spaCy pipeline and applied it to a sample text. It's that simple.

Note that we use spaCy's "blank" NLP pipeline here.
It actually carries a lot of information,
and defines spaCy's language-dependent, rule-based tokenizer.

!!! note "Non-destructive processing"

    In EDS-NLP, just like spaCy, non-destructiveness is a core principle.
    Your detected entities will **always** be linked to the **original text**.

    In other words, `#!python nlp(text).text == text` is always true.

### The `Doc` abstraction

The `doc` object carries the result of the entire processing.
It's the most important abstraction in spaCy,
and holds a token-based representation of the text along with the results of every pipeline components.
It also keeps track of the input text in a non-destructive manner, meaning that
`#!python doc.text == text` is always true.

```python
# ↑ Omitted code above ↑

# Text processing in spaCy is non-destructive
doc.text == text  # (1)

# You can access a specific token
token = doc[2]  # (2)

# And create a Span using slices
span = doc[:3]  # (3)

# Entities are tracked in the ents attribute
doc.ents  # (4)
# Out: ()
```

1.  This feature is a core principle in spaCy. It will always be true in EDS-NLP.
2.  `token` is a [`Token`](https://spacy.io/api/token) object referencing the third token
3.  `span` is a [`Span`](https://spacy.io/api/span) object referencing the first three tokens.
4.  We have not declared any entity recognizer in our pipeline, hence this attribute is empty.

### Adding pipeline components

You can add pipeline components with the `#!python nlp.add_pipe` method. Let's add two simple components to our pipeline.

```python hl_lines="5-6"
import spacy

nlp = spacy.blank("fr")

nlp.add_pipe("eds.sentences")  # (1)
nlp.add_pipe("eds.dates")  # (2)

text = "Le 5 mai 2005, Jimothé a été invité à une fête organisée par Michel."

doc = nlp(text)
```

1. Like the name suggests, this pipeline is declared by EDS-NLP.
   `eds.sentences` is a rule-based sentence boundary prediction.
   See [its documentation](../pipelines/core/sentences.md) for detail.
2. Like the name suggests, this pipeline is declared by EDS-NLP.
   `eds.dates` is a date extraction and normalisation component.
   See [its documentation](../pipelines/misc/dates.md) for detail.

The `doc` object just became more interesting!

```python
# ↑ Omitted code above ↑

# We can split the document into sentences
list(doc.sents)  # (1)
# Out: [Le 5 mai 2005, Jimothé a été invité à une fête organisée par Michel.]

# And look for dates
doc.spans["dates"]  # (2)
# Out: [5 mai 2005]

span = doc.spans["dates"][0]  # (3)
span._.date.to_datetime()  # (4)
# Out: DateTime(2005, 5, 5, 0, 0, 0, tzinfo=Timezone('Europe/Paris'))
```

1. In this example, there is only one sentence...
2. The `eds.dates` adds a key to the `doc.spans` attribute
3. `span` is a spaCy `Span` object.
4. In spaCy, you can declare custom extensions that live in the `_` attribute.
   Here, the `eds.dates` pipeline uses a `Span._.date` extension to persist the normalised date.
   We use the `to_datetime()` method to get an object that is usable by Python.

## Conclusion

This page is just a glimpse of a few possibilities offered by spaCy. To get a sense of what spaCy can help you achieve,
we **strongly recommend** you visit their [documentation](https://spacy.io/)
and take the time to follow the [spaCy course](https://course.spacy.io/en/).

Moreover, be sure to check out [spaCy's own crash course](https://spacy.io/usage/spacy-101){target="\_blank"}, which is an excellent read.
It goes into more detail on what's possible with the library.
