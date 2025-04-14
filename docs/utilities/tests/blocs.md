# Testing Code Blocs

We created a utility that scans through the documentation, extracts code blocs and executes them to check that everything is indeed functional.

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

We can disable code checking for a specific code bloc by adding a `.no-check` class to the code bloc:

````md
```python { .no-check }
test = undeclared_function(42)
```
````

Visit the source code of [test_docs.py](https://github.com/aphp/edsnlp/blob/master/tests/test_docs.py) for more information.
