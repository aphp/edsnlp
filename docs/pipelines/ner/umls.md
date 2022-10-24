# UMLS

The `eds.umls` pipeline component matches the UMLS (Unified Medical Language System from NIH) terminology.

!!! warning "Very low recall"

    When using the `exact' matching mode, this component has a very poor recall performance.
    We can use the `simstring` mode to retrieve approximate matches, albeit at the cost of a significantly higher computation time.

## Usage

`eds.umls` is an additional module that needs to be setup by:

1. `pip install -U umls_downloader`
2. [Signing up for a UMLS Terminology Services Account](https://uts.nlm.nih.gov/uts/signup-login). After filling a short form, you will receive your token API within a few days.
3. Set `UMLS_API_KEY` locally: `export UMLS_API_KEY=your_api_key`

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.umls")

text = "Grosse toux: le malade a été mordu par des Amphibiens " "sous le genou"

doc = nlp(text)

doc.ents
# Out: (toux, a, par, Amphibiens, genou)

ent = doc.ents[0]

ent.label_
# Out: umls

ent._.umls
# Out: C0010200
```

You can easily change the default languages and sources with the `pattern_config` argument:

```python
import spacy

# Enable the french and english languages, through the french MeSH and LOINC
pattern_config = dict(languages=["FRE", "ENG"], sources=["MSHFRE", "LNC"])

nlp = spacy.blank("fr")
nlp.add_pipe("eds.umls", config=dict(pattern_config=pattern_config))
```

See more options of languages and sources [here](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html).

## Configuration

The pipeline can be configured using the following parameters :

| Parameter             | Description                                                    | Default   |
|-----------------------|----------------------------------------------------------------|-----------|
| `term_matcher`        | Which algorithm should we use : `exact` or `simstring`         | `"exact"` |
| `term_matcher_config` | Config of the algorithm (`SimstringMatcher`'s for `simstring`) | `{}` |
| `pattern_config`      | Config of the terminology patterns loader                      | `{"languages"=["FRE"], sources=None}` (sources None means all available sources) |
| `attr`                | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)       | `"LOWER"` |
| `ignore_excluded`     | Whether to ignore excluded tokens for matching                 | `False`   |

## Authors and citation

The `eds.umls` pipeline was developed by AP-HP's Data Science team and INRIA SODA's team.
