# Tutorials

We provide step-by-step guides to get you started. We cover the following use-cases:

- [Matching a terminology](./matching-a-terminology.md): you're looking for a concept within a corpus of texts.
- [Qualifying entities](./qualifying-entities.md): you want to make sure that the concept you've extracted are not invalidated by linguistic modulation.
- [Detecting dates](./detecting-dates.md), which could serve as the basis for an event ordering algorithm.
- [Processing multiple texts](./multiple-texts.md): to improve the inference speed of your pipeline !

## Rationale

In a typical medical NLP pipeline, a group of clinicians would define a list of synonyms for a given concept of interest (say, for example, diabetes), and look for that terminology in a corpus of documents.

Now, consider the following example:

=== "French"

    ```
    Le patient n'est pas diabétique.
    Le patient est peut-être diabétique.
    Le père du patient est diabétique.
    ```

=== "English"

    ```
    The patient is not diabetic.
    The patient could be diabetic.
    The patient's father is diabetic.
    ```

There is an obvious problem: none of these examples should lead us to include this particular patient into the cohort.

To curb this issue, EDS-NLP proposes rule-based pipelines that qualify entities to help the user make an informed decision about which patient should be included in a real-world data cohort.

To sum up, a typical medical NLP project consists in:

1. Editing a terminology
2. "Matching" this terminology on a corpus, ie extract phrases that belong to that terminology
3. "Qualifying" entities to avoid false positives

Once the pipeline is ready, we need to deploy it efficiently.
