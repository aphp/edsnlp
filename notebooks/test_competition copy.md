---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: dldiy
    language: python
    name: python3
---

```python
import pandas as pd
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

from edsnlp.utils.flashtext import KeywordProcessor

import random
import string
import time
import spacy

from matplotlib import pyplot as plt
```

```python
data = pd.read_table('../data/drug.target.interaction.tsv')
data = data["DRUG_NAME"]
data = data.drop_duplicates()
data = data.reset_index()
data = data["DRUG_NAME"]
```

```python
len(data)
```

```python
def get_word_of_length(str_length):
    # generate a random word of given length
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(str_length))
```

```python
data_words = list(data)
text_length = 10000-len(data)
all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for i in range(text_length)]
tab_SimString = []
tab_FlashText = []
tab_FTspaCy = []
tab_EDSmatcher = []
print("EXACT COMPETITION")
print('Count | SimString | FlashText | FTspaCy | EDSmatcher |')
print('------------------------------------------------------')
for keywords_length in range(0, len(data), 500):
    all_words_chosen = all_words + data_words
    random.shuffle(all_words_chosen)
    story = ', '.join(all_words_chosen)

    # get unique keywords from the list of words generated.
    unique_keywords_sublist = list(set(random.sample(data_words, keywords_length)))

    # construct simstring
    db = DictDatabase(CharacterNgramFeatureExtractor(3))
    for medoc in unique_keywords_sublist :
        db.add(medoc)
    searcher = Searcher(db, CosineMeasure())


    terms = dict(
    medicament=list(unique_keywords_sublist),
    )
    # construct flashtext
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(unique_keywords_sublist)
    
    #construct flashtext spaCy
    nlp_flash = spacy.blank("fr")
    nlp_flash.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=0))


    # construct eds matcher
    nlp_eds = spacy.blank("fr")
    nlp_eds.add_pipe("eds.matcher", config=dict(terms=terms, ignore_excluded = False))

    # time the modules
    start = time.time()
    for query in all_words_chosen :
        _ = searcher.search(query, 1)
    mid1 = time.time()
    _ = keyword_processor.extract_keywords(story, max_cost=0)
    mid2 = time.time()
    _ = nlp_flash(story)
    mid3 = time.time()
    _ = nlp_eds(story)
    end = time.time()

    #updates tab
    tab_SimString.append(mid1 - start)
    tab_FlashText.append(mid2 - mid1)
    tab_FTspaCy.append(mid3 - mid2)
    tab_EDSmatcher.append(end - mid3)

    # print output
    print(str(keywords_length).ljust(5), '|',
          "{0:.5f}".format(mid1 - start).ljust(9), '|',
          "{0:.5f}".format(mid2 - mid1).ljust(9), '|',
          "{0:.5f}".format(mid3 - mid2).ljust(7), '|',
          "{0:.5f}".format(end - mid3).ljust(10), '|',)

plt.plot(range(0, len(data), 500),tab_SimString,label="SimString")
plt.plot(range(0, len(data), 500),tab_FlashText,label="FlashText")
plt.plot(range(0, len(data), 500),tab_FTspaCy,label="FlashTextspaCy")
plt.plot(range(0, len(data), 500),tab_EDSmatcher,label="EDS.matcher")
plt.legend()
plt.show()
```

```python
data_words = list(data)
text_length = 10000-len(data)
all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for i in range(text_length)]
tab_SimString = []
tab_FlashText = []
tab_FTspaCy = []
print("FUZZY COMPETITION")
print('Count | SimString | FlashText | FTspaCy |')
print('-----------------------------------------')
for keywords_length in range(0, len(data), 500):
    all_words_chosen = all_words + data_words
    random.shuffle(all_words_chosen)
    story = ', '.join(all_words_chosen)

    # get unique keywords from the list of words generated.
    unique_keywords_sublist = list(set(random.sample(data_words, keywords_length)))

    # construct simstring
    db = DictDatabase(CharacterNgramFeatureExtractor(3))
    for medoc in unique_keywords_sublist :
        db.add(medoc)
    searcher = Searcher(db, CosineMeasure())


    terms = dict(
    medicament=list(unique_keywords_sublist),
    )
    # construct flashtext
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(unique_keywords_sublist)
    
    #construct flashtext spaCy
    nlp_flash = spacy.blank("fr")
    nlp_flash.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=1))


    # time the modules
    start = time.time()
    for query in all_words_chosen :
        _ = searcher.search(query, 0.8)
    mid1 = time.time()
    _ = keyword_processor.extract_keywords(story, max_cost=1)
    mid2 = time.time()
    _ = nlp_flash(story)
    end = time.time()

    #updates tab
    tab_SimString.append(mid1 - start)
    tab_FlashText.append(mid2 - mid1)
    tab_FTspaCy.append(end - mid2)

    # print output
    print(str(keywords_length).ljust(5), '|',
          "{0:.5f}".format(mid1 - start).ljust(9), '|',
          "{0:.5f}".format(mid2 - mid1).ljust(9), '|',
          "{0:.5f}".format(end - mid2).ljust(7), '|')

plt.plot(range(0, len(data), 500),tab_SimString,label="SimString")
plt.plot(range(0, len(data), 500),tab_FlashText,label="FlashText")
plt.plot(range(0, len(data), 500),tab_FTspaCy,label="FlashTextspaCy")
plt.legend()
plt.show()
```

```python
data_words = list(data)
keywords_length = len(data)
tab_SimString = []
tab_FlashText = []
tab_FTspaCy = []
tab_EDSmatcher = []
print("EXACT COMPETITION")
print('Count | SimString | FlashText | FTspaCy | EDSmatcher |')
print('------------------------------------------------------')
for text_length in range(10000, 100000, 10000):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for i in range(text_length)]
    all_words_chosen = all_words + data_words
    random.shuffle(all_words_chosen)
    story = ', '.join(all_words_chosen)

    # get unique keywords from the list of words generated.
    unique_keywords_sublist = list(set(random.sample(data_words, keywords_length)))

    # construct simstring
    db = DictDatabase(CharacterNgramFeatureExtractor(3))
    for medoc in unique_keywords_sublist :
        db.add(medoc)
    searcher = Searcher(db, CosineMeasure())


    terms = dict(
    medicament=list(unique_keywords_sublist),
    )
    # construct flashtext
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(unique_keywords_sublist)
    
    #construct flashtext spaCy
    nlp_flash = spacy.blank("fr")
    nlp_flash.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=0))


    # construct eds matcher
    nlp_eds = spacy.blank("fr")
    nlp_eds.add_pipe("eds.matcher", config=dict(terms=terms, ignore_excluded = False))

    # time the modules
    start = time.time()
    for query in all_words_chosen :
        _ = searcher.search(query, 1)
    mid1 = time.time()
    _ = keyword_processor.extract_keywords(story, max_cost=0)
    mid2 = time.time()
    _ = nlp_flash(story)
    mid3 = time.time()
    _ = nlp_eds(story)
    end = time.time()

    #updates tab
    tab_SimString.append(mid1 - start)
    tab_FlashText.append(mid2 - mid1)
    tab_FTspaCy.append(mid3 - mid2)
    tab_EDSmatcher.append(end - mid3)

    # print output
    print(str(text_length).ljust(5), '|',
          "{0:.5f}".format(mid1 - start).ljust(9), '|',
          "{0:.5f}".format(mid2 - mid1).ljust(9), '|',
          "{0:.5f}".format(mid3 - mid2).ljust(7), '|',
          "{0:.5f}".format(end - mid3).ljust(10), '|',)

plt.plot(range(10000, 100000, 10000),tab_SimString,label="SimString")
plt.plot(range(10000, 100000, 10000),tab_FlashText,label="FlashText")
plt.plot(range(10000, 100000, 10000),tab_FTspaCy,label="FlashTextspaCy")
plt.plot(range(10000, 100000, 10000),tab_EDSmatcher,label="EDS.matcher")
plt.legend()
plt.show()
```

```python
data_words = list(data)
keywords_length = len(data)
tab_SimString = []
tab_FlashText = []
tab_FTspaCy = []
print("FUZZY COMPETITION")
print('Count | SimString | FlashText |  FTspaCy  |')
print('-------------------------------------------')
for text_length in range(10000, 100000, 10000):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for i in range(text_length-len(data))]
    all_words_chosen = all_words + data_words
    random.shuffle(all_words_chosen)
    story = ', '.join(all_words_chosen)

    # get unique keywords from the list of words generated.
    unique_keywords_sublist = list(set(random.sample(data_words, keywords_length)))

    # construct simstring
    db = DictDatabase(CharacterNgramFeatureExtractor(3))
    for medoc in unique_keywords_sublist :
        db.add(medoc)
    searcher = Searcher(db, CosineMeasure())


    terms = dict(
    medicament=list(unique_keywords_sublist),
    )
    # construct flashtext
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(unique_keywords_sublist)
    
    #construct flashtext spaCy
    nlp_flash = spacy.blank("fr")
    nlp_flash.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=1))


    # time the modules
    start = time.time()
    for query in all_words_chosen :
        _ = searcher.search(query, 0.8)
    mid1 = time.time()
    _ = keyword_processor.extract_keywords(story, max_cost=1)
    mid2 = time.time()
    _ = nlp_flash(story)
    end = time.time()

    #updates tab
    tab_SimString.append(mid1 - start)
    tab_FlashText.append(mid2 - mid1)
    tab_FTspaCy.append(end - mid2)

    # print output
    print(str(text_length).ljust(5), '|',
          "{0:.5f}".format(mid1 - start).ljust(9), '|',
          "{0:.5f}".format(mid2 - mid1).ljust(9), '|',
          "{0:.5f}".format(end - mid2).ljust(9), '|')

plt.plot(range(10000, 100000, 10000),tab_SimString,label="SimString")
plt.plot(range(10000, 100000, 10000),tab_FlashText,label="FlashText")
plt.plot(range(10000, 100000, 10000),tab_FTspaCy,label="FlashTextspaCy")
plt.plot(range(10000, 100000, 10000),tab_EDSmatcher,label="EDS.matcher")
plt.legend()
plt.show()
```

```python
all_words_chosen = all_words + data_words
random.shuffle(all_words_chosen)
story = ', '.join(all_words_chosen)

# get unique keywords from the list of words generated.
unique_keywords_sublist = list(set(random.sample(data_words, keywords_length)))

# construct simstring
db = DictDatabase(CharacterNgramFeatureExtractor(3))
for medoc in unique_keywords_sublist :
    db.add(medoc)
searcher = Searcher(db, CosineMeasure())


terms = dict(
medicament=list(unique_keywords_sublist),
)
# construct flashtext
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(unique_keywords_sublist)

#construct flashtext spaCy
nlp_flash = spacy.blank("fr")
nlp_flash.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=0))
```

```python
flash_result = keyword_processor.extract_keywords(story, max_cost = 0)
flash_result_span = keyword_processor.extract_keywords(story,span_info = True, max_cost = 0)
sim_result = []
for query in all_words_chosen :
        query_res = searcher.search(query, 1)
        if len(query_res)>0 :
            sim_result.append(query_res[0])
nlp_flash_result = list(nlp_flash(story).ents)
```

```python
len(all_words_chosen)
```

```python
len(unique_keywords_sublist)
```

```python
print(len(sim_result))
```

```python
print(len(flash_result))
```

```python
print(len(nlp_flash_result))
```

```python
for truc in flash_result :
    print(truc)
    break
```

```python
print(nlp_flash_result[3],nlp_flash_result[4])
```

```python
flash_result[3]
```

```python
flash_result_span[3]
```

```python
story[624-20:642+20]
```

```python
flash_result_span[4]
```

```python
a = 0
for truc in nlp_flash_result :
    if truc.text not in flash_result :
        print(truc, a)
    a += 1
```

```python
truc = []
double_truc = []
for word in data :
    if word not in truc :
        truc.append(word)
    else :
        double_truc.append(word)
len(double_truc)
```

```python
truc = []
double_truc = []
for span in nlp_flash_result :
    if span.text not in truc :
        truc.append(span.text)
    else :
        double_truc.append(span.text)
len(double_truc)
```

```python
double = []
for span in nlp_flash_result :
    if span.text in double_truc :
        double.append(span.text)
double
```
