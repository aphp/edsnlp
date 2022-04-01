# Sentences

The `eds.sentences` pipeline provides an alternative to spaCy's default `sentencizer`, aiming to overcome some of its limitations.

Indeed, the `sentencizer` merely looks at period characters to detect the end of a sentence, a strategy that often fails in a clinical note settings. Our `sentences` component also classifies end-of-lines as sentence boundaries if the subsequent token begins with an uppercase character, leading to slightly better performances.

Moreover, the `eds.sentences` pipeline can use the output of the `eds.normalizer` pipeline, and more specifically the end-of-line classification. This is activated by default.

## Usage

=== "EDS-NLP"

    <!-- no-check -->

    ```python
    import spacy

    nlp = spacy.blank("fr")
    nlp.add_pipe("eds.sentences")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur à l'estomac\n"
        "Il lui était arrivé la même chose il y a deux ans."
    )

    doc = nlp(text)

    for sentence in doc.sents:
        print("<s>", sentence, "</s>")
    # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
    # Out:  <\s>
    # Out: <s> Il lui était arrivé la même chose il y a deux ans. <\s>
    ```

=== "spaCy sentencizer"

    <!-- no-check -->

    ```python
    import spacy

    nlp = spacy.blank("fr")
    nlp.add_pipe("sentencizer")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur à l'estomac\n"
        "Il lui était arrivé la même chose il y a deux ans."
    )

    doc = nlp(text)

    for sentence in doc.sents:
        print("<s>", sentence, "</s>")
    # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
    # Out: Il lui était arrivé la même chose il y a deux ans. <\s>
    ```

Notice how EDS-NLP's implementation is more robust to ill-defined sentence endings.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                             | Default                           |
| -------------- | ----------------------------------------------------------------------- | --------------------------------- |
| `punct_chars`  | Punctuation patterns                                                    | `None` (use pre-defined patterns) |
| `use_endlines` | Whether to use endlines prediction (see [documentation](./endlines.md)) | `True`                            |

## Authors and citation

The `eds.sentences` pipeline was developed by AP-HP's Data Science team.
