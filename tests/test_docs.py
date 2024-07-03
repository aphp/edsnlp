import sys

import pytest

pytest.importorskip("mkdocs")

from extract_docs_code import extract_docs_code  # noqa: E402

# We don't check documentation for Python <= 3.7:
if sys.version_info < (3, 8):
    url_to_code = {}
else:
    url_to_code = dict(extract_docs_code())


def printer(code: str) -> None:
    """
    Prints a code bloc with lines for easier debugging.

    Parameters
    ----------
    code : str
        Code bloc.
    """
    lines = []
    for i, line in enumerate(code.split("\n")):
        lines.append(f"{i + 1:03}  {line}")

    print("\n".join(lines))


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("url", sorted(url_to_code.keys()), ids=str)
def test_code_blocks(url):
    raw = url_to_code[url]
    try:
        exec(raw, {"__MODULE__": "__main__"})
    except Exception:
        printer(raw)
        raise
