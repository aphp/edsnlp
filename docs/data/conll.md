# CoNLL

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    stream = edsnlp.data.read_conll(path)
    stream = stream.map_pipeline(nlp)
    ```

You can easily integrate CoNLL formatted files into your project by using EDS-NLP's CoNLL reader.

There are many CoNLL formats corresponding to different shared tasks, but one of the most common is the CoNLL-U format, which is used for dependency parsing. In CoNLL files, each line corresponds to a token and contains various columns with information about the token, such as its index, form, lemma, POS tag, and dependency relation.

EDS-NLP lets you specify the name of the `columns` if they are different from the default CoNLL-U format. If the `columns` parameter is unset, the reader looks for a comment containing `# global.columns` to infer the column names. Otherwise, the columns are

```
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
```

A typical CoNLL file looks like this:

```{ title="sample.conllu" }
1	euh	euh	INTJ	_	_	5	discourse	_	SpaceAfter=No
2	,	,	PUNCT	_	_	1	punct	_	_
3	il	lui	PRON	_	Gender=Masc|Number=Sing|Person=3|PronType=Prs	5	expl:subj	_	_
...
```

## Reading CoNLL files {: #edsnlp.data.conll.read_conll }

::: edsnlp.data.conll.read_conll
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
