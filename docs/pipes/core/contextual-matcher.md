# Contextual Matcher {: #edsnlp.pipes.core.contextual_matcher.factory.create_component }

EDS-NLP provides simple pattern matchers like `eds.matcher` to extract regular expressions, specific phrases, or perform lexical similarity matching on documents. However, certain use cases require examining the context around matched entities to filter out irrelevant matches or enrich them with additional information. For example, to extract mentions of malignant cancers, we need to exclude matches that have “benin” mentioned nearby : `eds.contextual_matcher` was built to address such needs.

## Example

```python
import edsnlp, edsnlp.pipes as eds

nlp = edsnlp.blank("eds")

nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(
    eds.contextual_matcher(
        patterns=[
            dict(
                terms=["cancer", "tumeur"],  # (1)!
                regex=[r"adeno(carcinom|[\s-]?k)", "neoplas", "melanom"],  # (2)!
                regex_attr="NORM",  # (3)!
                exclude=dict(
                    regex="benign|benin",  # (4)!
                    window=3,  # (5)!
                ),
                assign=[
                    dict(
                        name="stage",  # (6)!
                        regex="stade (I{1,3}V?|[1234])",  # (7)!
                        window="words[-10:10]",  # (8)!
                        replace_entity=False,  # (9)!
                        reduce_mode=None,  # (10)!
                    ),
                    dict(
                        name="metastase",  # (11)!
                        regex="(metasta)",  # (12)!
                        window=10,  # (13)!
                        replace_entity=False,  # (14)!
                        reduce_mode="keep_last",  # (15)!
                    ),
                ],
            ),
            dict(
                source="Lymphome",  # (16)!
                regex=["lymphom", "lymphangio"],  # (17)!
                regex_attr="NORM",  # (18)!
                exclude=dict(
                    regex=["hodgkin"],  # (19)!
                    window=3,  # (20)!
                ),
            ),
        ],
        label="cancer",
    ),
)
```

1. Exact match terms (faster than regex, but less flexible)
2. Regex for flexible matching
3. Apply regex on normalized text
4. Regex to exclude benign mentions
5. Window size for exclusion check
6. Extract cancer stage
7. Stage regex pattern
8. Window range for stage extraction. Visit the documentation of [ContextWindow][edsnlp.utils.span_getters.ContextWindow] for more information about this syntax.
9. Do not use these matches as replacement for the anchor (default behavior)
10. Keep all matches
11. Detect metastasis
12. Regex for metastasis detection
13. Window size for detection
14. Keep main entity
15. Keep furthest extraction
16. Source label for lymphoma
17. Regex patterns for lymphoma
18. Apply regex on normalized text
19. Exclude Hodgkin lymphoma
20. Window size for exclusion

Let's explore some examples using this pipeline:

=== "Simple match"

    ```python
    txt = "Le patient a eu un cancer il y a 5 ans"
    doc = nlp(txt)
    ent = doc.ents[0]

    ent.label_
    # Out: cancer

    ent._.source
    # Out: Cancer solide

    ent.text, ent.start, ent.end
    # Out: ('cancer', 5, 6)
    ```

=== "Exclusion rule"

    Check exclusion with a benign mention:

    ```python
    txt = "Le patient a eu un cancer relativement bénin il y a 5 ans"
    doc = nlp(txt)

    doc.ents
    # Out: ()
    ```

=== "Extracting additional infos"

    Additional information extracted via `assign` configurations is available in the `assigned` attribute:

    ```python
    txt = "Le patient a eu un cancer de stade 3."
    doc = nlp(txt)

    doc.ents[0]._.assigned
    # Out: {'stage': '3'}
    ```

## Better control over the final extracted entities

Three main parameters refine how entities are extracted:

#### `include_assigned`

Following the previous example, if you want extracted entities to include the cancer stage or metastasis status (if found), set `include_assigned=True` in the pipe configuration.

For instance, from the sentence "Le patient a un cancer au stade 3":

- If `include_assigned=False`, the extracted entity is "cancer"
- If `include_assigned=True`, the extracted entity is "cancer au stade 3"

#### `reduce_mode`

Sometimes, an assignment matches multiple times. For example, in the sentence "Le patient a un cancer au stade 3 et au stade 4", both "stade 3" and "stade 4" match the `stage` key. Depending on your use case:

- `reduce_mode=None` (default): Keeps all matched extractions in a list
- `reduce_mode="keep_first"`: Keeps only the extraction closest to the main matched entity ("stade 3" in this case)
- `reduce_mode="keep_last"`: Keeps only the furthest extraction

#### `replace_entity`

This parameter can be set to `True` **for only one assign key per dictionary**. If set to `True`, the matched assignment replaces the main entity.

Example using "Le patient a un cancer au stade 3":

- With `replace_entity=True` for the `stage` key, the entity extracted is "stade 3"
- With `replace_entity=False`, the entity extracted remains "cancer"

**Note**: With `replace_entity=True`, if the corresponding assign key matches nothing, the entity is discarded.

The primary configuration is provided in the `patterns` key as either a **pattern dictionary** or a **list of pattern dictionaries**.

::: edsnlp.pipes.core.contextual_matcher.factory.create_component
    options:
        only_parameters: true

## Authors and citation

The `eds.contextual_matcher` pipeline component was developed by AP-HP's Data Science team.
