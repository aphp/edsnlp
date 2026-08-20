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

The trade-off is deliberate, but the value space is not open. The pattern has
always been the gate — the previous one restricted stages just as narrowly
(`[0-4o]|is` for T, `[0-3o]|x` for N, `[01o]|x` for M) and the enums merely
duplicated that check in the model. Only the duplicate is gone; the pattern
still constrains every stage to a closed set.

| Field        | Accepted values                     |
|--------------|-------------------------------------|
| `tumour`     | `0`-`4`, `is`, `x`                  |
| `node`       | `0`-`4`, `x`, `+`                   |
| `metastasis` | `0`-`3`, `x`, `+`                   |
| `pleura`     | `0`-`3`, `x`                        |
| `resection`  | `0`, `1`, `2`, `x`, `+`             |

So `pT2N50M0` is not matched, and neither are `N5`, `M4`, `T7` or `R5`. What
the enums could not represent — `N4`, `M2`, `M3`, `Rx`, `R+`, metastasis site
codes such as `PUL` — is accepted now. Only the free-text fields are genuinely
open: `*_suffix`, `resection_loc`, and node ratios such as `(3/12)`. Enforce
your own constraints if you build `TNM` instances by hand rather than through
the pipe.

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

## Evaluation {: #evaluation }

The pipe was qualified by two physicians before production use, on an initial sample of 20 million
clinical notes stratified by year, restricted to the ~5 million documents
belonging to patients followed for cancer.

Sampling used Neyman allocation over strata, with a minimum of 5 documents per
stratum, and the reported figures are the corresponding stratum-weighted
estimates.

| Metric                     | Estimate  | Confidence interval | Unit      | Sample |
|----------------------------|-----------|---------------------|-----------|--------|
| Precision                  | 98.64 %   | ± 1 % (95 % CI)     | mention   | 366    |
| Recall (entity level)      | 79.40 %   | ± 1 % (99 % CI)     | mention   | 120    |
| Recall (document level)    | 95.53 %   | ± 1 % (99 % CI)     | document  | 120    |

Document-level recall is the share of documents containing at least one TNM
mention for which at least one mention is retrieved. It is much higher than the
entity-level figure because staging is usually repeated within a report.

Strata were built on the document type (pathology report / multidisciplinary meeting report
vs. other), oncological activity of the care unit measured with the prevalence of cancer related ICD10 codes, and — for
precision — whether the mention carried only a T component, which is the
configuration most prone to false positives. Recall strata additionally split on
whether the pipe found a mention and whether the raw text contained the word
`TNM`.

**Precision — error analysis.** 21 false positives out of 366 annotations. Most
come from the rule accepting the letter `o` as a substitute for the digit `0`
(`p t o m`, `TOM`, `TOC`, `CS tox`). The rest are interfering acronyms (`CMT1A`,
`RT 3D`), numbering or temporal wording (`Tour 1`, `au T1`), and mentions whose
format is a valid TNM but whose context is not (`T1 (bloc n°3)`).

**Recall — error analysis.** 27 false negatives, matching the patterns listed in
the "Known limitations" warning above.
