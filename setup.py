import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlptools",
    version="0.0.1",
    author="Data Science - DSI APHP",
    author_email="basile.dura-ext@aphp.fr",
    description="NLP tools for human consumption at AP-HP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)