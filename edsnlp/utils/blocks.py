import inspect
import re
import textwrap
from pathlib import Path
from typing import List

BLOCK_PATTERN = re.compile(
    r"((?P<skip><!-- no-check -->)\s+)?```(?P<title>.*?)\n(?P<code>.+?)```",
    flags=re.DOTALL,
)


def get_codeblock_members(*classes):
    """
    Grabs the docstrings of any methods of any classes that are passed in.
    """
    results = []
    for cl in classes:
        if cl.__doc__:
            results.append(cl)
        for name, member in inspect.getmembers(cl):
            if member.__doc__:
                results.append(member)
    return [m for m in results if len(grab_code_blocks(m.__doc__)) > 0]


def check_outputs(block):

    lines: List[str] = block.split("\n")
    code = []

    skip = False

    if len(lines) < 2:
        return block

    for expression, output in zip(lines[:-1], lines[1:]):
        if skip:
            skip = not skip
            continue

        if output.startswith("# Out: "):
            expression = f"v = {expression}"

            output = output[len("# Out: ") :].replace('"', r"\"")
            output = f'assert repr(v) == "{output}" or str(v) == "{output}"'

            code.append(expression)
            code.append(output)

            skip = True

        else:
            code.append(expression)

    if not skip:
        code.append(output)

    return "\n".join(code)


def check_codeblock(block, lang="python"):
    """
    Cleans the found codeblock and checks if the proglang is correct.

    Returns an empty string if the codeblock is deemed invalid.

    Arguments:
        block: the code block to analyse
        lang: if not None, the language that is assigned to the codeblock
    """
    first_line = block.split("\n")[0]

    if lang:
        if not first_line[3:].startswith(lang):
            return ""
    code = "\n".join(block.split("\n")[1:])
    code = check_outputs(code)
    return code


def grab_code_blocks(docstring, lang="python"):
    """
    Given a docstring, grab all the markdown codeblocks found in docstring.

    Arguments:
        docstring: the docstring to analyse
        lang: if not None, the language that is assigned to the codeblock
    """
    docstring = textwrap.dedent(docstring)
    codeblocks = []

    for match in BLOCK_PATTERN.finditer(docstring):
        d = match.groupdict()

        if d["skip"]:
            continue

        if lang in d["title"]:
            code = check_outputs(d["code"])
            codeblocks.append(code)

    return codeblocks


def printer(block: str):
    lines = []
    for i, line in enumerate(block.split("\n")):
        lines.append(f"{i + 1:02}  {line}")

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


def check_md_file(path, memory=False):
    """
    Given a markdown file, parse the contents for python code blocks
    and check that each independant block does not cause an error.

    Arguments:
        path: path to markdown file
        memory: wheather or not previous code-blocks should be remembered
    """
    text = Path(path).read_text()
    if memory:
        check_raw_file_full(text, lang="python")
    else:
        check_raw_string(text, lang="python")
