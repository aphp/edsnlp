from setuptools import find_packages, setup

import edsnlp


def get_lines(relative_path):
    with open(relative_path) as f:
        return f.readlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="edsnlp",
    version=edsnlp.__version__,
    author="Data Science - DSI APHP",
    author_email="basile.dura-ext@aphp.fr",
    description="NLP tools for human consumption at AP-HP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=get_lines("requirements.txt"),
    extras_require=dict(
        demo=["streamlit>=1.2"],
        spark=["pyspark"],
    ),
    package_data={
        "edsnlp": ["resources/*.csv"],
    },
)
