from itertools import chain
from pathlib import Path

import pytest

from edsnlp.utils.blocks import check_md_file

files = chain(
    Path("./").glob("*.md"),
    Path("docs").glob("**/*.md"),
)


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("path", files, ids=str)
def test_code_blocks(path):
    check_md_file(path=path, memory=True)
