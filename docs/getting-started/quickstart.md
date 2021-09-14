# Quickstart

EDS-NLP declares easy-to-use Spacy pipelines. Just import `edsnlp`, load a Spacy pipeline, add components and start extracting data from French medical notes !

```python
import spacy

# Load declared pipelines
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe(
    "matcher",
    config=dict(regex=dict(covid=r"covid[\-\s]?(?:19)?|corona[\-\s]?virus")),
)
nlp.add_pipe("negation")

text = "Le patient n'est pas atteint de la covid-19"

doc = nlp(text)

doc.ents
# Out: (covid-19,)

doc.ents[0]._.polarity_
# Out: 'NEG'
```

Let's unpack what happened :

1. We start by declaring a bare-bones spacy pipeline.

   ```python
   nlp = spacy.blank("fr")
   ```

2. Then, we add three pipes : a [sentencizer](../user-guide/sentences.md), a [matcher](../user-guide/matcher.md) and a [negation detector](../user-guide/negation.md).

   ```python
   nlp.add_pipe("sentences")
   nlp.add_pipe(
       "matcher",
       config=dict(regex=dict(covid=r"covid[\-\s]?(?:19)?|corona[\-\s]?virus")),
   )
   nlp.add_pipe("negation")
   ```

   As the name suggests, the sentencizer detects sentence boundaries and populates the `doc.sents` attribute.

   The matcher extracts entities based on a dictionary of terms or regular expressions. Here, the component looks for synonyms of covid.

   The `negation` components performs negation detection on matched entities, to limit false positives.

3. We defined a text and the pipeline to it.

   ```python
   text = "Le patient n'est pas atteint de la covid-19"

   doc = nlp(text)
   ```

4. We look at the entities discovered by the matcher :

   ```python
   doc.ents
   # Out: (covid-19,)
   ```

5. We make sure that the negation was picked up by the `negation` component :
   ```python
   doc.ents[0]._.polarity_
   # Out: 'NEG'
   ```

This was just a very simple example, using only three of the numerous components defined by EDS-NLP. To get a better sense of what EDS-NLP can do for you, check out the user guides or the tutorials.
