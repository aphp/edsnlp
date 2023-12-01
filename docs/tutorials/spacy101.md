# SpaCy representations

EDS-NLP uses spaCy to represent documents and their annotations. You will need to familiarise yourself with some key spaCy concepts.

!!! tip "Skip if you're familiar with spaCy objects"

    This page is intended as a crash course for the very basic spaCy concepts that are needed to use EDS-NLP. If you've already used spaCy, you should probably skip to the next page.

## The `Doc` object

The `doc` object carries the result of the entire processing.
It's the most important abstraction in spaCy, hence its use in EDS-NLP, and holds a token-based representation of the text along with the results of every pipeline components. It also keeps track of the input text in a non-destructive manner, meaning that
`#!python doc.text == text` is always true.

To obtain a doc, run the following code:
```python
import edsnlp  # (1)

# Initialize a pipeline
nlp = edsnlp.blank("eds")  # (2)

text = "Michel est un penseur latéral."  # (3)

# Apply the pipeline
doc = nlp(text)  # (4)

doc.text
# Out: 'Michel est un penseur latéral.'

# If you do not want to run the pipeline but only tokenize the text
doc = nlp.make_doc(text)

# Text processing in spaCy is non-destructive
doc.text == text

# You can access a specific token
token = doc[2]  # (5)

# And create a Span using slices
span = doc[:3]  # (6)

# Entities are tracked in the ents attribute
doc.ents  # (7)
# Out: ()
```

1. Import edsnlp...
2. Load a pipeline. The parameter corresponds to the [language](/tokenizers) code and affects the tokenization.
3. Define a text you want to process.
4. Apply the pipeline and get a spaCy [`Doc`](https://spacy.io/api/doc) object.
5. `token` is a [`Token`](https://spacy.io/api/token) object referencing the third token
6. `span` is a [`Span`](https://spacy.io/api/span) object referencing the first three tokens.
7. We have not declared any entity recognizer in our pipeline, hence this attribute is empty.

We just created a pipeline and applied it to a sample text. It's that simple.

## The `Span` objects

Span of text are represented by the `Span` object and represent slices of the `Doc` object. You can either create a span by slicing a `Doc` object, or by running a pipeline component that creates spans. There are different types of spans:

- `doc.ents` are non-overlapping spans that represent entities
- `doc.sents` are the sentences of the document
- `doc.spans` is dict of groups of spans (that can overlap)

```python
import edsnlp

nlp = edsnlp.blank("eds")

nlp.add_pipe("eds.sentences")  # (1)
nlp.add_pipe("eds.dates")  # (2)

text = "Le 5 mai 2005, Jimothé a été invité à une fête organisée par Michel."

doc = nlp(text)
```

1. Like the name suggests, this pipeline component is declared by EDS-NLP.
   `eds.sentences` is a rule-based sentence boundary prediction.
   See [its documentation](/pipes/core/sentences) for detail.
2. Like the name suggests, this pipeline component is declared by EDS-NLP.
   `eds.dates` is a date extraction and normalisation component.
   See [its documentation](/pipes/misc/dates) for detail.

The `doc` object just became more interesting!

```{ .python .no-check }
# ↑ Omitted code above ↑

# We can split the document into sentences spans
list(doc.sents)  # (1)
# Out: [Le 5 mai 2005, Jimothé a été invité à une fête organisée par Michel.]

# And list dates spans
doc.spans["dates"]  # (2)
# Out: [5 mai 2005]

span = doc.spans["dates"][0]  # (3)
```

1. In this example, there is only one sentence...
2. The `eds.dates` adds a key to the `doc.spans` attribute
3. `span` is a spaCy `Span` object.

## SpaCy extensions

We can add custom attributes (or "extensions") to spaCy objects via the `_` attribute. For example, the `eds.dates` pipeline adds a `Span._.date` extension to the `Span` object. The attributes can be any Python object.

```{ .python .no-check }
# ↑ Omitted code above ↑

span._.date.to_datetime()  # (1)
# Out: DateTime(2005, 5, 5, 0, 0, 0, tzinfo=Timezone('Europe/Paris'))
```

1. We use the `to_datetime()` method of the extension to get an object that is usable by Python.
