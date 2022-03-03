# Demo

To get a glimpse of what EDS-NLP can do for you, run the interactive demo !

<!-- termynal -->

```
# Clone the repo
$ git clone https://github.com/aphp/edsnlp.git
---> 100%

# Move to the repo directory
$ cd edsnlp

# Optionally create an environment and activate it
$ python -m venv .venv && source .venv/bin/activate

# Install the project with the demo requirements
$ pip install '.[demo]'
---> 100%

# Run the demo
$ streamlit run scripts/demo.py
```

Go to [`localhost:8501`](http://localhost:8501){target="\_blank"} to see the library in action.

!!! warning

    The above code will not work within JupyterLab. You need to execute it locally.
