import ast
import inspect
import re
import sys
import textwrap
import warnings
from math import isclose

import catalogue
import pytest
from spacy.tokens.underscore import Underscore

pytest.importorskip("mkdocs")
try:
    import torch.nn
except ImportError:
    torch = None

if torch is None:
    pytest.skip("torch not installed", allow_module_level=True)
pytest.importorskip("rich")

from extract_docs_code import extract_docs_code  # noqa: E402

# We don't check documentation for Python <= 3.7:
if sys.version_info < (3, 8):
    url_to_code = {}
else:
    url_to_code = dict(extract_docs_code())
    # just to make sure something didn't go wrong
    assert len(url_to_code) > 50


class nested_approx:
    def __init__(self, value, rel=1e-12, abs=1e-12):
        self._value, self._rel, self._abs = value, rel, abs

    def __eq__(self, other):
        return self._match(self._value, other)

    def __req__(self, other):
        return self._match(other, self._value)

    __hash__ = None  # keep it un-hashable

    def _match(self, a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return isclose(a, b, rel_tol=self._rel, abs_tol=self._abs)
        if isinstance(a, (list, tuple)):
            return (
                isinstance(b, (list, tuple))
                and len(a) == len(b)
                and all(self._match(x, y) for x, y in zip(a, b))
            )
        if isinstance(a, dict):
            return (
                isinstance(b, dict)
                and a.keys() == b.keys()
                and all(self._match(a[k], b[k]) for k in a)
            )
        return a == b

    def __repr__(self):
        return f"nested_approx({self._value!r}, rel={self._rel}, abs={self._abs})"


pytest.nested_approx = nested_approx


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


def insert_assert_statements(code):
    line_table = [0]
    for line in code.splitlines(keepends=True):
        line_table.append(line_table[-1] + len(line))

    tree = ast.parse(code)
    replacements = []

    for match in re.finditer(
        r"^\s*#\s*Out\s*: (.*$(?:\n#\s.*$)*)", code, flags=re.MULTILINE
    ):
        lineno = code[: match.start()].count("\n")
        for stmt in tree.body:
            if stmt.end_lineno == lineno:
                if isinstance(stmt, ast.Expr):
                    expected = textwrap.dedent(match.group(1)).replace("\n# ", "\n")
                    expected_s = expected
                    begin = line_table[stmt.lineno - 1]
                    if not (expected.startswith("'") or expected.startswith('"')):
                        expected_s = repr(expected)
                    end = match.end()
                    stmt_str = ast.unparse(stmt)
                    if stmt_str.startswith("print("):
                        stmt_str = stmt_str[len("print") :]
                    repl = f"""\
val = {stmt_str}
try:
    import ast
    expected = ast.literal_eval({expected_s})
except (ValueError, SyntaxError):
    expected = None
assert str(val) == {expected_s} or val == pytest.nested_approx(expected, 0.01, 0.01)
"""
                    replacements.append((begin, end, repl))
                if isinstance(stmt, ast.For):
                    expected = textwrap.dedent(match.group(1)).split("\n# Out: ")
                    expected = [line.replace("\n# ", "\n") for line in expected]
                    begin = line_table[stmt.lineno - 1]
                    end = match.end()
                    stmt_str = ast.unparse(stmt).replace("print", "assert_print")
                    repl = f"""\
printed = []
{stmt_str}
assert printed == {expected}
"""
                    replacements.append((begin, end, repl))

    for begin, end, repl in reversed(replacements):
        code = code[:begin] + repl + code[end:]

    return code


# TODO: once in a while, it can be interesting to run reset_imports for each code block,
#  instead of only once and tests should still pass, but it's way slower.
@pytest.fixture(scope="module")
def reset_imports():
    """
    Reset the imports for each test.
    """
    # 1. Clear registered functions to avoid using cached ones
    for k, m in list(catalogue.REGISTRY.items()):
        mod = inspect.getmodule(m)
        if mod is not None and mod.__name__.startswith("edsnlp"):
            del catalogue.REGISTRY[k]

    # Let's ensure that we "bump" into every possible warnings:
    # 2. Remove all modules that start with edsnlp, to reimport them
    for k in list(sys.modules):
        if k.split(".")[0] == "edsnlp":
            del sys.modules[k]

    # 3. Delete spacy extensions to avoid error when re-importing
    Underscore.span_extensions.clear()
    Underscore.doc_extensions.clear()
    Underscore.token_extensions.clear()


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("url", sorted(url_to_code.keys()), ids=str)
def test_code_blocks(url, tmpdir, reset_imports):
    code = url_to_code[url]
    code_with_asserts = """
import pytest

def assert_print(*args, sep=" ", end="\\n", file=None, flush=False):
    printed.append((sep.join(map(str, args)) + end).rstrip('\\n'))

""" + insert_assert_statements(code)
    assert "# Out:" not in code_with_asserts, (
        "Unparsed asserts in {url}:\n" + code_with_asserts
    )
    # We'll import test_code_blocks from here
    sys.path.insert(0, str(tmpdir))
    test_file = tmpdir.join("test_code_blocks.py")

    # Clear all warnings
    warnings.resetwarnings()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings(module=".*endlines.*", action="ignore")
            warnings.filterwarnings(
                message="__package__ != __spec__.parent", action="ignore"
            )
            # First, forget test_code_blocks
            sys.modules.pop("test_code_blocks", None)

            # Then, reimport it, to let pytest do its assertion rewriting magic
            test_file.write_text(code_with_asserts, encoding="utf-8")

            import test_code_blocks  # noqa: F401

            exec(
                compile(code_with_asserts, test_file, "exec"),
                {"__MODULE__": "__main__"},
            )
    except ImportError as e:
        # skip UMLS tests if the package is not installed
        if "umls_downloader" in str(e):
            pytest.skip("umls_downloader package not found")
        else:
            raise
    except Exception:
        printer(code_with_asserts)
        raise
