# Display single text outputs

If you are

- Developping a new pipeline
- Testing various inputs on an existing pipeline
- ...

you might want to quickly apply a pipeline and display the output `doc` in a comprehensible way.

 ```{ .python .no-check }
from edsnlp.viz import QuickExample

E = QuickExample(nlp)  # (1)
```

1. This is the `Language` instance that should be defined beforehands

Next, simply call `E` with any string:

 ```{ .python .no-check }
txt = "Le patient présente une anomalie."
E(txt)
```

<div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_html rendered_html" dir="auto"><pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                              Le </span><span style="font-weight: bold; font-style: italic">patient </span><span style="font-style: italic">présente une </span><span style="font-weight: bold; font-style: italic">anomalie</span><span style="font-style: italic">                               </span>
<span style="font-style: italic">                                                                                             </span>
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Entity   </span>┃<span style="font-weight: bold"> Source   </span>┃<span style="font-weight: bold"> eds.hypoth… </span>┃<span style="font-weight: bold"> eds.negation </span>┃<span style="font-weight: bold"> eds.family </span>┃<span style="font-weight: bold"> eds.history </span>┃<span style="font-weight: bold"> eds.report… </span>┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ patient  │ patient  │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │ <span style="color: #800000; text-decoration-color: #800000">False</span>        │ <span style="color: #800000; text-decoration-color: #800000">False</span>      │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │
│ anomalie │ anomalie │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │ <span style="color: #800000; text-decoration-color: #800000">False</span>        │ <span style="color: #800000; text-decoration-color: #800000">False</span>      │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │ <span style="color: #800000; text-decoration-color: #800000">False</span>       │
└──────────┴──────────┴─────────────┴──────────────┴────────────┴─────────────┴─────────────┘
</pre>
</div>

By default, each `Qualifiers` in `nlp` adds a corresponding column to the output. Additionnal informations can be displayed by using the `extensions` parameter. For instance, if entities have a custom `ent._.custom_ext` extensions, it can be displayed by providing the extension when instantiating `QuickExample`:

 ```{ .python .no-check }
E = QuickExample(nlp, extensions=["_.custom_ext"])
```

Finally, if you prefer to output a DataFrame instead of displaying a table, set the `as_dataframe` parameter to True:

 ```{ .python .no-check }
E = QuickExample(nlp)
E(txt, as_dataframe=True)
```
