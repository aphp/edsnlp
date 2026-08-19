# TNM {: #edsnlp.pipes.ner.tnm.factory.create_component }

::: edsnlp.pipes.ner.tnm.factory.create_component
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Migrating from the previous `eds.tnm` {: #migration }

The regex and the `TNM` model were rewritten. Extracted spans and `norm()`
values are broadly compatible, but **the model fields changed**, so any code
reading `span._.tnm.<field>` needs updating.

### Renamed fields

| Before                    | Now                | Note                                                     |
|---------------------------|--------------------|----------------------------------------------------------|
| `prefix`                  | `tumour_prefix`    | Each component now carries its own prefix                 |
| `resection_completeness`  | `resection`        | Was an `int`, is now a `str`                              |

```{ .python .no-check }
# Before
tnm = doc.ents[0]._.tnm
tnm.prefix, tnm.resection_completeness

# Now
tnm = doc.ents[0]._.tnm
tnm.tumour_prefix, tnm.resection
```

### Enums replaced by strings

`Prefix`, `Tumour`, `Specification`, `Node`, `Metastasis` and `TnmEnum` were
removed from `edsnlp.pipes.ner.tnm.model`. Every field now holds the raw
matched text as a `str` (except `version_year`, an `int`).

```{ .python .no-check }
# Before -- fields were enum members
from edsnlp.pipes.ner.tnm.model import Tumour
tnm.tumour is Tumour.score_2
str(tnm.tumour) == "2"

# Now -- fields are plain strings
tnm.tumour == "2"
```

The trade-off is deliberate: the new pattern accepts values the enums could not
represent (`N4`, `M2`, `M3`, `Rx`, `R+`, site codes such as `PUL`), so values
are no longer validated against a closed set. Validate downstream if your use
case requires it.

### New fields

`node_prefix`, `metastasis_prefix`, `resection_prefix`,
`metastasis_specification`, `metastasis_suffix`, `resection_specification`,
`resection_loc`, `resection_suffix` and `pleura`. They default to `None`, so
existing code keeps working — but `norm()` and `span.kb_id_` now include them,
which means a normalised value may be longer than before for the same text.

### Suffixes in `norm()`

`norm()` used to concatenate `tumour_suffix` and `node_suffix` verbatim, so a
parenthesised comment ended up inside `span.kb_id_` (`pT1(grade 2)N1M0` gave
`pT1grade 2N1M0`). Only suffixes that read as a TNM qualifier — one to three
letters, such as the `(m)` of `pT1(m)` — are carried into `norm()` now. The
full text remains available on the field.

### Behaviour changes to be aware of

- **Matching is case-insensitive.** `pt2n1m0` and `PT2N1M0` are now extracted;
  previously only certain case combinations were.
- **A lone T is no longer extracted** unless it carries both a prefix and a
  specification (`pT2b`), or is followed by an N, M or R component. `pT2` and
  `pTx` used to match and no longer do. This is the main driver of the
  precision gain.
- **`o` → `0` coercion is now restricted** to the numeric stage fields, so
  `M1OSS` keeps its `O`. Previously every field was coerced.
- **A component prefix binds to its own component.** In `pT1 cN1 M0` the `c`
  is now `node_prefix`; it used to land in `tumour_specification`. `norm()` is
  unchanged for this input.
- **`banned_words`** is a new parameter. Pass an empty list to restore the
  unfiltered regex output.

### Patterns module

`patterns_new.py` was merged into `patterns.py` and the old pattern removed.
Import `tnm_pattern` from `edsnlp.pipes.ner.tnm.patterns`; `tnm_pattern_new`
no longer exists.
