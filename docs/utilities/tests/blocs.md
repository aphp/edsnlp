# Testing Code Blocs

We created a utility that scans through markdown files, extracts code blocs and executes them to check that everything is indeed functional.

There is more! Whenever the utility comes across an example (denoted by `# Out: `, see example below), an `assert` statement is dynamically added to the snippet to check that the output matches.

For instance:

```python
a = 1

a
# Out: 1
```

Is transformed into:

```python
a = 1

v = a
assert repr(v) == "1"
```

We can disable code checking for a specific code bloc by adding `<!-- no-check -->` above it:

````md
<!-- no-check -->

```python
test = undeclared_function(42)
```
````

See the [dedicated reference][edsnlp.utils.blocks.check_md_file] for more information
