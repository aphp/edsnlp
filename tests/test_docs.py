from pathlib import Path

import pytest

from edsnlp.utils.blocks import check_md_file


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("path", Path("docs").glob("**/*.md"), ids=str)
def test_code_blocks(path):
    check_md_file(path=path, memory=True)
