<!-- --8<-- [start:pipe-definition] -->

```python
text = """Poids : 65. Taille : 1.75
          On mesure ... à 3mmol/l ; pression : 100mPa-110mPa.
          Acte réalisé par ... à 12h13"""
```

=== "All measurements"

    ```python
    import edsnlp
    
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.tables")
    nlp.add_pipe(
        "eds.measurements",
        config=dict(measurements="all", 
                    extract_ranges=True, # (1)
                    use_tables=True), # (2)
    )
    nlp(text).spans["measurements"]
    # Out: [65, 1.75, 3mmol/l, 100mPa-110mPa, 12h13]
    ```

    1. 100-110mg, 2 à 4 jours ...
    2. If True `eds.tables` must be called

=== "Custom measurements"

    ```python
    import edsnlp
    
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.tables")
    nlp.add_pipe(
        "eds.measurements",
        config=dict(measurements={"concentration": {"unit": "mol_per_l"},
                                  "pressure": {"unit": "Pa"}},
                    extract_ranges=True, # (1)
                    use_tables=True), # (2)
    )
    nlp(text).spans["measurements"]
    # Out: [3mmol/l, 100mPa-110mPa]
    ```

    1. 100-110mg, 2 à 4 jours ...
    2. If True `eds.tables` must be called

=== "Predefined measurements"

    ```python
    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.tables")
    nlp.add_pipe(
        "eds.measurements",
        config=dict(measurements=["weight", "size"],
                    extract_ranges=True, # (1)
                    use_tables=True), # (2)
    )
    nlp(text).spans["measurements"]
    # Out: [65, 1.75]
    ```

    1. 100-110mg, 2 à 4 jours ...
    2. If True `eds.tables` must be called

<!-- --8<-- [end:pipe-definition] -->

<!-- --8<-- [start:availability_mes_units] -->

See `edsnlp.pipes.misc.measurements.patterns` for exhaustive definition.

=== "Available measurements"

    | measurement_name | Example                |
    |------------------|------------------------|
    | `size`           | `1m50`, `1.50m`...     |
    | `weight`         | `12kg`, `1kg300`...    |
    | `bmi`            | `BMI: 24`, `24 kg.m-2` |
    | `volume`         | `2 cac`, `8ml`...      |
    
=== "Available units"

    | units                          | Example                |
    |--------------------------------|------------------------|
    | `mass`                         | `10kgr`, `100mg`...    |
    | `mole`                         | `10mmol`, `3mol`...    |
    | `length`, `surface`, `volume`  | `12m`, `1cm2`, 0.1l... |
    | `time`                         | `1h`,  `2min15`...     |
    | `pressure`                     | `1kPa`, `100mmHg`...   |
    | `temperature`                  | `5°C`, `100mmHg`...    |
    | `count`                        | `2.2x10*2` ...         |

<!-- --8<-- [end:availability_mes_units] -->
