# Word embeddings

EDS-NLP proposes rule-based components exclusively. However, that does not prohibit you from exploiting SpaCy's machine learning capabilities!
You can mix and match machine learning pipelines, trainable or not, with EDS-NLP rule-based components.

In this tutorial, we will explore how you can use **static word vectors** trained with [Gensim](https://radimrehurek.com/gensim/) within SpaCy.

Training the word embedding, however, is outside the scope of this post. You'll find very well designed resources on the subject in [Gensim's documenation](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py).

!!! tip "Using Transformer models"

    SpaCy v3 introduced support for Transformer models through their helper library `spacy-transformers` that interfaces with
    HuggingFace's `transformers` library.

    Using transformer models can significantly increase your model's performance.

## Adding pre-trained word vectors

SpaCy provides a `init vectors` CLI utility that takes a Gensim-trained binary and transforms it to a SpaCy-readable pipeline.

Using it is straightforward :

<!-- termynal -->

```
$ spacy init vectors fr /path/to/vectors /path/to/pipeline
---> 100%
Conversion successful!
```

See the [documentation](https://spacy.io/api/cli#init-vectors) for implementation details.
