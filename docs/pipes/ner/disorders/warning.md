!!! danger "On overlapping entities"
    When using multiple disorders or behavior pipelines, some entities may be extracted from different pipes. For instance:

    * "Intoxication éthylotabagique" will be tagged both by `eds.tobacco` and `eds.alcohol`
    * "Chirrose alcoolique" will be tagged both by `eds.liver_disease` and `eds.alcohol`

    As `doc.ents` discards overlapping entities, you should use `doc.spans` instead.
