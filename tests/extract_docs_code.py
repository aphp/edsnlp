import re
import shutil
import tempfile
from textwrap import dedent
from typing import Tuple

from markdown.extensions import Extension
from markdown.extensions.attr_list import get_attrs
from markdown.extensions.codehilite import parse_hl_lines
from markdown.extensions.fenced_code import FencedBlockPreprocessor
from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.config.config_options import Type as MkType
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin
from mkdocstrings.extension import AutoDocProcessor
from mkdocstrings.plugin import MkdocstringsPlugin

BRACKET_RE = re.compile(r"\[([^\[]+)\]")
CITE_RE = re.compile(r"@([\w_:-]+)")
DEF_RE = re.compile(r"\A {0,3}\[@([\w_:-]+)\]:\s*(.*)")
INDENT_RE = re.compile(r"\A\t| {4}(.*)")

CITATION_RE = r"(\[@(?:[\w_:-]+)(?: *, *@(?:[\w_:-]+))*\])"


class PyCodePreprocessor(FencedBlockPreprocessor):
    """Gather reference definitions and citation keys"""

    FENCED_BLOCK_RE = re.compile(
        dedent(
            r"""
            (?P<fence>^[ ]*(?:~{3,}|`{3,}))[ ]*                          # opening fence
            ((\{(?P<attrs>[^\}\n]*)\})|                              # (optional {attrs} or
            (\.?(?P<lang>[\w#.+-]*)[ ]*)?                            # optional (.)lang
            (hl_lines=(?P<quot>"|')(?P<hl_lines>.*?)(?P=quot)[ ]*)?) # optional hl_lines)
            \n                                                       # newline (end of opening fence)
            (?P<code>.*?)(?<=\n)                                     # the code block
            (?P=fence)[ ]*$                                          # closing fence
        """  # noqa: E501
        ),
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

    def __init__(self, md, code_blocks):
        super().__init__(md, {})
        self.code_blocks = code_blocks

    def run(self, lines):
        text = "\n".join(lines)
        if 'nlp.add_pipe(f"eds.aids")' in text:
            print("TEXT", text)
        while True:
            # ----  https://github.com/Python-Markdown/markdown/blob/5a2fee/markdown/extensions/fenced_code.py#L84C9-L98  # noqa: E501
            m = self.FENCED_BLOCK_RE.search(text)
            if 'nlp.add_pipe(f"eds.aids")' in text:
                print("CODE ==>", m.group("code") if m else None)
            if m:
                lang, id, classes, config = None, "", [], {}
                if m.group("attrs"):
                    id, classes, config = self.handle_attrs(get_attrs(m.group("attrs")))
                    if len(classes):
                        lang = classes.pop(0)
                else:
                    if m.group("lang"):
                        lang = m.group("lang")
                    if m.group("hl_lines"):
                        # Support `hl_lines` outside of `attrs` for
                        # backward-compatibility
                        config["hl_lines"] = parse_hl_lines(m.group("hl_lines"))
                # ----
                code = m.group("code")

                if lang == "python" and "no-check" not in classes:
                    self.code_blocks.append(dedent(code))
            else:
                break
            text = text[m.end() :]

        return lines


context_citations = None


class PyCodeExtension(Extension):
    def __init__(self, code_blocks):
        super(PyCodeExtension, self).__init__()
        self.code_blocks = code_blocks

    def extendMarkdown(self, md):
        self.md = md
        md.registerExtension(self)
        md.preprocessors.register(
            PyCodePreprocessor(md, self.code_blocks), "fenced_code", 31
        )
        for ext in md.registeredExtensions:
            if isinstance(ext, AutoDocProcessor):
                ext._config["mdx"].append(self)


def makeExtension(*args, **kwargs):
    return PyCodeExtension(*args, **kwargs)


class PyCodeExtractorPlugin(BasePlugin):
    config_scheme: Tuple[Tuple[str, MkType]] = (
        # ("bibtex_file", MkType(str)),  # type: ignore[assignment]
        # ("order", MkType(str, default="unsorted")),  # type: ignore[assignment]
    )

    def __init__(self, global_config):
        self.global_config = global_config
        self.page_code_blocks = []
        self.docs_code_blocks = []

    def on_config(self, config: MkDocsConfig):
        self.ext = PyCodeExtension(self.page_code_blocks)
        # After pymdownx.highlight, because of weird registering deleting the first
        # extension
        config["markdown_extensions"].append(self.ext)
        config["markdown_extensions"].remove("pymdownx.highlight")
        config["markdown_extensions"].remove("fenced_code")

    def on_pre_build(self, *, config: MkDocsConfig):
        mkdocstrings_plugin: MkdocstringsPlugin = config.plugins["mkdocstrings"]
        mkdocstrings_plugin.get_handler("python")

    def on_page_content(self, html, page, config, files):
        if len(self.page_code_blocks):
            self.docs_code_blocks.append((page.url, "\n".join(self.page_code_blocks)))
        self.page_code_blocks.clear()
        return html


def extract_docs_code():
    config = load_config()

    temp_dir = tempfile.mkdtemp()
    try:
        config["site_dir"] = temp_dir

        # plug the pycode extractor plugin
        plugin = PyCodeExtractorPlugin(config)
        config.plugins["pycode_extractor"] = plugin

        config["plugins"].run_event("startup", command="build", dirty=False)
        try:
            build(config)
        finally:
            config["plugins"].run_event("shutdown")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return plugin.docs_code_blocks
