---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.2"
      jupytext_version: 1.6.0
  kernelspec:
    display_name: "[2.4.3] Py3"
    language: python
    name: pyspark-2.4.3
---

# Using and speeding-up EDS-NLP

The way EDS-NLP is used may depend on how many documents you are working with. Once working with tens of thousands of them,
parallelizing the processing can be really efficient (up to 8x faster), but will require a (tiny) bit more work.
Here are shown 3 ways to analyse texts depending on your needs:

- [Testing / Using on a single string](#1.-Pipeline-on-a-single-string)
- [Using on a few documents](#2.-Pipeline-on-a-few-documents)
- [Using on many documents](#3.-Pipeline-on-many-documents)

```python
import spacy
import pandas as pd

import time
from datetime import timedelta

from tqdm import tqdm

# One-shot import of all declared Spacy components
import edsnlp.components

# Module containing processing helpers
import edsnlp.multiprocessing.processing as nlprocess
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('sentences')
nlp.add_pipe('normalizer')

terms = dict(covid=["coronavirus", "covid19", "covid"])

nlp.add_pipe("matcher", config=dict(terms=terms, attr="NORM"))
nlp.add_pipe('negation')
nlp.add_pipe('hypothesis')
nlp.add_pipe('family')
```

## 1. Pipeline on a single string

```python
text = """
    Patient admis pour suspicion de Covid.
    Pas de cas de coronavirus dans ce service.
    Le père du patient est atteind du covid.
"""
```

Simply apply `nlp()` to the piece of text:

```python
doc = nlp(text)
```

We can have a quick look at what was extracted here:

```python
def pretty_ents_printer(ents, limit=5):

    headers = "{:<15} {:<20} {:<30} {:<6} {:<6} {:<6}"

    print (headers.format('Text', 'Label','Span','Neg','Par','Hyp'))
    for entite in ents[:limit]:
        print(headers.format(entite.text,
                             entite.label_,
                             f"({entite.start_char},{entite.end_char})",
                             entite._.negated,
                             entite._.family,
                             entite._.hypothesis))
```

```python
pretty_ents_printer(doc.ents)
```

## 2. Pipeline on a few documents

We will here get documents from the cluster. Depending on your acces, change the following parameters:

```python
DB_NAME = "edsomop_prod_a"
TABLE_NAME = "orbis_note"
NOTE_ID_COL = "note_id"
NOTE_TEXT_COL = "note_text"
```

```python
notes = sql(
    f"""
    SELECT
        {NOTE_ID_COL} AS note_id,
        {NOTE_TEXT_COL} AS note_text
    FROM
        {DB_NAME}.{TABLE_NAME}
    WHERE
        {NOTE_TEXT_COL} IS NOT NULL
    LIMIT 100000
    """
).toPandas()
```

Let us keep 1000 documents to make a small set of notes

```python
small_notes_subset = notes[:1000]
```

Using the `nlprocess.pipe` method (see its documentation for more details), we can directly give the DataFrame as input.
A SpaCy document will be created from each line.
If entities are extracted, we will store them in a list:

```python
ents = []
for doc in nlprocess.pipe(nlp,
                          small_notes_subset,
                          text_col='note_text',
                          context_cols=['note_id'],
                          progress_bar=False):
    if len(doc.ents) > 0:
        ents.extend(list(doc.ents))
```

```python
pretty_ents_printer(ents, limit=15)
```

## 3. Pipeline on many documents

To go even faster, we have to **parallelize** the task.

For more details, check the documentation in `Tutorials - Getting faster`

To sum up what changes when parallelizing:

1. The task is broken up into multiple processes.
2. Each process saves intermediary results on memory.
3. At the end, those results are aggregated and returned.

The step 2. imposes that the intermediary results are **serializable**, i.e. we cannot simply save the SpaCy `Doc` object.
We need to tell the pipe what to save for each document: it is the goal of the `pick_results` function defined here:

```python
def pick_results(doc):
    """
    Function used well Paralellizing tasks via joblib
    This functions will store all extracted entities
    """
    return [{'note_id':e.doc._.note_id,
             'lexical_variant':e.text,
             'offset_start':e.start_char,
             'offset_end':e.end_char,
             'label':e.label_,
             'negation':e._.negated,
             'family':e._.family,
             'hypothesis':e._.hypothesis} for e in doc.ents if doc.ents]
```

You can adjust this function however suits your needs the best.

Finally, the method `parallel_pipe` wraps everything up:

```python
ents = nlprocess.parallel_pipe(nlp,
                               notes,
                               chunksize=100,
                               n_jobs=-2,
                               context_cols='note_id',
                               progress_bar=False,
                               return_df=True,
                               pick_results = pick_results)
```

```python
ents.head()
```

By giving the `note_id` into the `context_cols` argument, you can easily merge the results with your input DataFrame and keep on with your analysis

```python
ents = ents.merge(notes, on='note_id', how='inner')
```

## 4. Time comparison

Let us compare the last 2 methods on various number of documents

```python
def process(notes, method):
    """
    Compare runtime between the two methods
    """
    n = len(notes)
    t0 = time.time()

    if method == "Single process":

        results = []
        for doc in nlprocess.pipe(nlp,
                                  notes,
                                  text_col='note_text',
                                  context_cols=['note_id'],
                                  progress_bar=False):
            if len(doc.ents) > 0:
                results.extend(list(doc.ents))

    elif method == "Parallel":

        results = nlprocess.parallel_pipe(nlp,
                                          notes,
                                          chunksize=100,
                                          n_jobs=-2,
                                          context_cols='note_id',
                                          progress_bar=False,
                                          return_df=True,
                                          pick_results = pick_results)

    t1 = round(time.time() - t0)
    str_time = str(timedelta(seconds=t1))
    speed = round(60*n/t1)

    print(f"{method}: Took {str_time} for {n} documents --> Mean of {speed} docs/minute")
```

```python
list_notes = [
    notes[:100],
    notes[:1000],
    notes[:10000]
]

list_methods = [
    "Single process", # 2. Pipeline on a few documents
    "Parallel"  # 3. Pipeline on many documents
]
```

```python
for notes_subset in list_notes:
    for method in list_methods:
        process(notes_subset, method)
```

We can see that while the parallel method has some overhead with a few hundreds of documents, it gets way quicker with the number of inputs increasing.
It can run on the full 100.000 documents fairly quickly:

```python
process(notes, "Parallel")
```
