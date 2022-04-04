"""
Utility that extracts code blocs and runs them.

Largely inspired by https://github.com/koaning/mktestdocs
"""

import re
from pathlib import Path
from typing import List

BLOCK_PATTERN = re.compile(
    (
        r"((?P<skip><!-- no-check -->)\s+)?(?P<indent> *)"
        r"```(?P<title>.*?)\n(?P<code>.+?)```"
    ),
    flags=re.DOTALL,
)
OUTPUT_PATTERN = "# Out: "


def check_outputs(code: str) -> str:
    """
    Looks for output patterns, and modifies the bloc:

    1. The preceding line becomes `#!python v = expr`
    2. The output line becomes an `#!python assert` statement

    Parameters
    ----------
    code : str
        Code block

    Returns
    -------
    str
        Modified code bloc with assert statements
    """

    lines: List[str] = code.split("\n")
    code = []

    skip = False

    if len(lines) < 2:
        return code

    for expression, output in zip(lines[:-1], lines[1:]):
        if skip:
            skip = not skip
            continue

        if output.startswith(OUTPUT_PATTERN):
            expression = f"v = {expression}"

            output = output[len(OUTPUT_PATTERN) :].replace('"', r"\"")
            output = f'assert repr(v) == "{output}" or str(v) == "{output}"'

            code.append(expression)
            code.append(output)

            skip = True

        else:
            code.append(expression)

    if not skip:
        code.append(output)

    return "\n".join(code)


def remove_indentation(code: str, indent: int) -> str:
    """
    Remove indentation from a code bloc.

    Parameters
    ----------
    code : str
        Code bloc
    indent : int
        Level of indentation

    Returns
    -------
    str
        Modified code bloc
    """

    if not indent:
        return code

    lines = []

    for line in code.split("\n"):
        lines.append(line[indent:])

    return "\n".join(lines)


def grab_code_blocks(docstring: str, lang="python") -> List[str]:
    """
    Given a docstring, grab all the markdown codeblocks found in docstring.

    Parameters
    ----------
    docstring : str
        Full text.
    lang : str, optional
        Language to execute, by default "python"

    Returns
    -------
    List[str]
        Extracted code blocks
    """
    codeblocks = []

    for match in BLOCK_PATTERN.finditer(docstring):
        d = match.groupdict()

        if d["skip"]:
            continue

        if lang in d["title"]:
            code = remove_indentation(d["code"], len(d["indent"]))
            code = check_outputs(code)
            codeblocks.append(code)

    return codeblocks


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


def check_docstring(obj, lang=""):
    """
    Given a function, test the contents of the docstring.
    """
    for b in grab_code_blocks(obj.__doc__, lang=lang):
        try:
            exec(b, {"__MODULE__": "__main__"})
        except Exception:
            print(f"Error Encountered in `{obj.__name__}`. Caused by:\n")
            printer(b)
            raise


def check_raw_string(raw, lang="python"):
    """
    Given a raw string, test the contents.
    """
    for b in grab_code_blocks(raw, lang=lang):
        try:
            exec(b, {"__MODULE__": "__main__"})
        except Exception:
            printer(b)
            raise


def check_raw_file_full(raw, lang="python"):
    all_code = "\n".join(grab_code_blocks(raw, lang=lang))
    try:
        exec(all_code, {"__MODULE__": "__main__"})
    except Exception:
        printer(all_code)
        raise


def check_md_file(path: Path, memory: bool = False) -> None:
    """
    Given a markdown file, parse the contents for Python code blocs
    and check that each independant bloc does not cause an error.

    Parameters
    ----------
    path : Path
        Path to the markdown file to execute.
    memory : bool, optional
        Whether to keep results from one bloc to the next, by default `#!python False`
    """
    text = Path(path).read_text()
    if memory:
        check_raw_file_full(text, lang="python")
    else:
        check_raw_string(text, lang="python")
