# Tokenizers


In addition to the standard spaCy `FrenchLanguage` (`fr`), EDS-NLP offers a new language better fit
for French clinical documents: `EDSLanguage` (`eds`). Additionally, the `EDSLanguage` document creation should be around 5-6 times faster than
the `fr` language. The main differences lie in the tokenization process.

A comparison of the two tokenization methods is demonstrated below:

| Example            | FrenchLanguage            | EDSLanguage                               |
|--------------------|---------------------------|-------------------------------------------|
| `ACR 5`            | \[`ACR5`\]                | \[`ACR`, `5`\]                            |
| `26.5/`            | \[`26.5/`\]               | \[`26.5`, `/`\]                           |
| `\n \n CONCLUSION` | \[`\n \n`, `CONCLUSION`\] | \[`\n`, `\n`, `CONCLUSION`\]              |
| `l'artère`         | \[`l'`, `artère`\]        | \[`l'`, `artère`\] (same)                 |
| `Dr. Pichon`       | \[`Dr`, `.`, `Pichon`\]   | \[`Dr.`, `Pichon`\]                       |
| `B.H.HP.A.7.A`     | \[`B.H.HP.A.7.A`\]        | \[`B.`, `H.`, `HP.`, `A`, `7`, `A`, `0`\] |

To instantiate one of the two languages, you can call the `spacy.blank` method.

=== "EDSLanguage"

    ```python
    import spacy

    nlp = spacy.blank("eds")
    ```

=== "FrenchLanguage"

    ```python
    import spacy

    nlp = spacy.blank("fr")
    ```
