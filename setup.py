from setuptools import setup, find_packages

def get_lines(relative_path):
    with open(relative_path) as f:
        return f.readlines()

INSTALL_REQUIRES = get_lines('requirements.txt')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlptools",
    version="0.0.3",
    author="Data Science - DSI APHP",
    author_email="basile.dura-ext@aphp.fr",
    description="NLP tools for human consumption at AP-HP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES
)
