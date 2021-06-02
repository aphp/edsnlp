# Testing the algorithm

Various tests for the components of the Spacy pipeline.

We decided to design tests entity-wise, meaning that we only check the validity
of the computed modality on a set of entities. This design choice is motivated by
the fact that :

1. That's what we actually care about. We want our pipeline to detect negation,
   family context, patient history and hypothesis relative to a given entity.

2. Deciding on the span of an annotation (negated, hypothesis, etc) is tricky.
   Consider the example : `"Le patient n'est pas malade."`. Should the negated span
   correspond to `["est", "malade"]`, `["malade"]`, `["n'", "est", "pas", "malade", "."]` ?

3. Depending on the design of the algorithm, the span might be off, even though it
   can correctly assign polarity to a given entity (but considered that the punctuation 
   was negated as well).
   By relaxing the need to infer the correct span, we avoid giving an unfair disadvantage
   to an otherwise great algorithm.
