"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

files = [
    "changelog.md",
    "contributing.md",
]

for f in files:
    path = Path(f)

    with mkdocs_gen_files.open(path, "w") as fd:
        fd.write(path.read_text())
