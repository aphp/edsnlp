# Matchers

We implemented two pattern matchers that are fit to clinical documents:

- the `EDSPhraseMatcher`
- the `RegexMatcher`

However, note that for most use-cases, you should instead use the `eds.matcher` pipeline that wraps these classes to annotate documents.

## EDSPhraseMatcher

The EDSPhraseMatcher lets you efficiently match large terminology lists, by comparing tokenx against a given attribute.
This matcher differs from the `spacy.PhraseMatcher` in that it allows to skip pollution tokens. To make it efficient, we
have reimplemented the matching algorithm in Cython, like the original `spacy.PhraseMatcher`.

You can use it as described in the code below.

```python
import spacy
from edsnlp.matchers.phrase import EDSPhraseMatcher

nlp = spacy.blank("eds")
nlp.add_pipe("eds.normalizer")
doc = nlp("On ne relève pas de signe du Corona =============== virus.")

matcher = EDSPhraseMatcher(nlp.vocab, attr="NORM")
matcher.build_patterns(
    nlp,
    {
        "covid": ["corona virus", "coronavirus", "covid"],
        "diabete": ["diabete", "diabetique"],
    },
)

list(matcher(doc, as_spans=True))[0].text
# Out: Corona =============== virus
```

## RegexMatcher

The `RegexMatcher` performs full-text regex matching.
It is especially useful to handle spelling variations like `mammo-?graphies?`.
Like the `EDSPhraseMatcher`, this class allows to skip pollution tokens.
Note that this class is significantly slower than the `EDSPhraseMatcher`: if you can, try enumerating
lexical variations of the target phrases and feed them to the `PhraseMatcher` instead.

You can use it as described in the code below.

```python
import spacy
from edsnlp.matchers.regex import RegexMatcher

nlp = spacy.blank("eds")
nlp.add_pipe("eds.normalizer")
doc = nlp("On ne relève pas de signe du Corona =============== virus.")

matcher = RegexMatcher(attr="NORM", ignore_excluded=True)
matcher.build_patterns(
    {
        "covid": ["corona[ ]*virus", "covid"],
        "diabete": ["diabete", "diabetique"],
    },
)

list(matcher(doc, as_spans=True))[0].text
# Out: Corona =============== virus
```
