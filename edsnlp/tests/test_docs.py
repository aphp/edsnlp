from itertools import chain
from pathlib import Path

import pytest

from edsnlp.utils.blocs import check_md_file

# @pytest.fixture(autouse=True, scope="module")
# def brat_folder():
#     yield
#     shutil.rmtree("path/to/brat")


files = chain(
    Path("./").glob("*.md"),
    Path("docs").glob("**/*.md"),
)


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("path", files, ids=str)
def test_code_blocks(path):
    check_md_file(path=path, memory=True)
