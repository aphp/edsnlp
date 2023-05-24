import os
import shutil
from pathlib import Path

import mkdocs

# Add the files from the project root

files = [
    "changelog.md",
    "contributing.md",
]

docs_gen = Path("docs")
os.makedirs(docs_gen, exist_ok=True)

for f in files:
    with open(docs_gen / Path(f), "w") as fd:
        fd.write(Path(f).read_text())

# Generate the code reference pages and navigation.
doc_reference = Path("docs/reference")
shutil.rmtree(doc_reference, ignore_errors=True)
os.makedirs(doc_reference, exist_ok=True)

for path in sorted(Path("edsnlp").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to("edsnlp").with_suffix(".md")
    full_doc_path = doc_reference / doc_path

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    ident = ".".join(parts)

    os.makedirs(full_doc_path.parent, exist_ok=True)
    with open(full_doc_path, "w") as fd:
        print(f"# `{ident}`\n", file=fd)
        print("::: " + ident, file=fd)


def on_files(files: mkdocs.structure.files.Files, config: mkdocs.config.Config) -> None:
    """
    Updates the navigation to take code reference files into account
    """
    reference_files = []
    for file in files:
        if file.src_path.startswith("reference/"):
            current = reference_files
            parts = ["edsnlp"] + file.src_path.replace(".md", "").split("/")[1:]
            for part in parts[:-1]:
                entry = next(
                    (
                        next(iter(entry.values()))
                        for entry in current
                        if next(iter(entry.keys())) == part
                    ),
                    None,
                )
                if entry is None:
                    entry = []
                    current.append({part: entry})
                    current = entry
                else:
                    current = entry
            current.append({parts[-1]: file.src_path})

    def rec(tree):
        if isinstance(tree, str) and tree.strip("/") == "reference":
            return reference_files
        elif isinstance(tree, list):
            return [rec(item) for item in tree]
        elif isinstance(tree, dict):
            return {k: rec(item) for k, item in tree.items()}
        else:
            return tree

    new_nav = rec(config["nav"])
    config["nav"] = new_nav
