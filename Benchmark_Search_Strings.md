# SimString

Search with a given measure (for example Cosine Similarity) if a string is in a bunch of string. Base of QuickUMLS.
Quite annoying to use (you have to split the text in order to have a list of query word/expressions).


# FlashText

Build a Tree with all the words in the dictionary, and go through it with the query tokens. Built to find exact match, maybe it's possible to refactor it to allow some "errors" in the crossing of the tree (it should not explode so much).

Possibilitie to allow some misspeling through the Levensthein's Distance, already implemented in flash text but not in the `pip` version, so the code is in `edsnlp\utils\flashtext.py`.


# Comparison between SimString and FlashText :

`Count` is the number of Keyword used, the search is in a text of 10000 words (all the keywords + random words)

On exact match : `SimString` is about 10 times slower than `FlashText`
```
Count  | FlashText | SimString |
-------------------------------
0      | 0.02000   | 0.18351   |
500    | 0.02801   | 0.26502   |
1000   | 0.02800   | 0.19202   |
1500   | 0.03500   | 0.26702   |
2000   | 0.03100   | 0.20246   |
2500   | 0.03399   | 0.20818   |
```

On fuzzy match (max 1 Levensthein dist for `FlashText` and 0.8 or cosine similarity for `SimString`) : `SimString` is about 2 times faster than `FlashText`

```
Count  | FlashText | SimString |
-------------------------------
0      | 0.06601   | 1.14901   |
500    | 1.76875   | 1.00130   |
1000   | 2.11886   | 1.01619   |
1500   | 2.27485   | 1.03588   |
2000   | 1.95617   | 1.11460   |
2500   | 2.13228   | 1.15821   |
```


# Global Comparison with SimString, FlashText (with and without a spaCy pipeline) and the EDS.matcher pipeline :

## Making the number of keywords evolve (we search them in 10k words)

```
EXACT COMPETITION
Count | SimString | FlashText | FTspaCy | EDSmatcher |
------------------------------------------------------
0     | 0.16901   | 0.01900   | 0.88706 | 0.78106    |
500   | 0.29502   | 0.02600   | 0.83106 | 1.32510    |
1000  | 0.28202   | 0.02800   | 0.78306 | 1.84014    |
1500  | 0.32902   | 0.03000   | 0.83706 | 2.64216    |
2000  | 0.21301   | 0.03200   | 0.81806 | 3.09923    |
2500  | 0.35903   | 0.03700   | 0.76006 | 3.44325    |
```

```
FUZZY COMPETITION
Count | SimString | FlashText | FTspaCy |
-----------------------------------------
0     | 1.16013   | 0.05101   | 0.89707 |
500   | 1.04108   | 2.24417   | 2.92714 |
1000  | 1.05508   | 2.32512   | 3.05924 |
1500  | 1.03408   | 3.09474   | 3.92729 |
2000  | 1.21609   | 2.06315   | 2.95722 |
2500  | 1.10708   | 2.16816   | 2.96522 |
```

## Making the number of total words evolve (we search 2587 keywords)


```
EXACT COMPETITION
Count | SimString | FlashText | FTspaCy | EDSmatcher |
------------------------------------------------------
10000 | 0.39403   | 0.04200   | 1.17209 | 4.73035    |
20000 | 0.59956   | 0.06701   | 1.83514 | 8.09860    |
30000 | 0.71705   | 0.08101   | 2.53819 | 12.29189   |
40000 | 0.94507   | 0.14701   | 4.00055 | 16.79729   |
```

```
FUZZY COMPETITION
Count | SimString | FlashText |  FTspaCy  |
-------------------------------------------
10000 | 1.35110   | 2.88821   | 4.91637   |
20000 | 2.21816   | 6.73250   | 7.80158   |
30000 | 2.33017   | 9.09268   | 12.11979  |
40000 | 3.31178   | 11.93089  | 14.07093  |
```




# QuickUmls

<span style="color:red">
Difficult to install
</span>

## General info

`https://github.com/Georgetown-IR-Lab/QuickUMLS`

Require a license from the National Library of Medicine + an installation using MetaprohoSys

```
"In this work, we introduce a system that relies on approximate matching to terms in UMLS to extract medical concepts from unstructured text. Our implementation is able to extract concepts using the entire English subset of UMLS from a document of approximately 1000 tokens2 within 500 to 1000 ms, depending on the threshold used for approximate string matching."
```

135 times faster than MetaMap and 25 times faster than cTakes, with similar F1-Score but we can choose the balance between Prec and Rec.

## spaCy

module spécial spaCy : SpacyQuickUMLS

```
nlp = spacy.load('en_core_web_sm')

quickumls_component = SpacyQuickUMLS(nlp, 'PATH_TO_QUICKUMLS_DATA')
nlp.add_pipe(quickumls_component)
```
=> easy integration



