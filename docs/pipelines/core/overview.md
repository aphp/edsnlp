# Core Components

This section deals with "core" functionalities offered by EDS-NLP:

- Generic matchers against regular expressions and list of terms
- Text cleaning
- Sentence boundaries detection

## Available components

<!-- --8<-- [start:components] -->

| Component                                                      | Description                                     |
|----------------------------------------------------------------|-------------------------------------------------|
| [`eds.normalizer`](/pipelines/core/normalizer)                 | Non-destructive input text normalisation        |
| [`eds.sentences`](/pipelines/core/sentences)                   | Better sentence boundary detection              |
| [`eds.matcher`](/pipelines/core/matcher)                       | A simple yet powerful entity extractor          |
| [`eds.terminology`](/pipelines/core/terminology)               | A simple yet powerful terminology matcher       |
| [`eds.contextual_matcher`](/pipelines/core/contextual-matcher) | A conditional entity extractor                  |
| [`eds.endlines`](/pipelines/core/endlines)                     | An unsupervised model to classify each end line |

<!-- --8<-- [end:components] -->
