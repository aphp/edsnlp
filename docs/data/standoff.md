# BRAT and Standoff

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    stream = edsnlp.data.read_standoff(path)
    stream = stream.map_pipeline(nlp)
    res = stream.write_standoff(path)
    # or equivalently
    edsnlp.data.write_standoff(stream, path)
    ```

You can easily integrate [BRAT](https://brat.nlplab.org/) into your project by using EDS-NLP's BRAT reader and writer.

BRAT annotations are in the [standoff format](https://brat.nlplab.org/standoff.html). Consider the following document:

```{ title="doc.txt" }
Le patient est admis pour une pneumopathie au coronavirus.
On lui prescrit du paracétamol.
```

Brat annotations are stored in a separate file formatted as follows:

```{ title="doc.ann" }
T1	Patient 4 11	patient
T2	Disease 31 58	pneumopathie au coronavirus
T3	Drug 79 90	paracétamol
```

## Reading Standoff files {: #edsnlp.data.standoff.read_standoff }

::: edsnlp.data.standoff.read_standoff
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Writing Standoff files {: #edsnlp.data.standoff.write_standoff }

::: edsnlp.data.standoff.write_standoff
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
