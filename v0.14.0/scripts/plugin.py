import os
from pathlib import Path

import jedi
import mkdocs.config
import mkdocs.plugins
import mkdocs.structure
import mkdocs.structure.files
import mkdocs.structure.nav
import mkdocs.structure.pages
from bs4 import BeautifulSoup


def exclude_file(name):
    return name.startswith("assets/fragments/")


# Add the files from the project root

VIRTUAL_FILES = {}
REFERENCE_TEMPLATE = """
# `{ident}`
::: {ident}
    options:
        show_source: false
"""


def on_files(files: mkdocs.structure.files.Files, config: mkdocs.config.Config):
    """
    Recursively the navigation of the mkdocs config
    and recursively content of directories of page that point
    to directories.

    Parameters
    ----------
    config: mkdocs.config.Config
        The configuration object
    kwargs: dict
        Additional arguments
    """

    root = Path("edsnlp")
    reference_nav = []
    for path in sorted(root.rglob("*.py")):
        module_path = path.relative_to(root.parent).with_suffix("")
        doc_path = Path("reference") / path.relative_to(root.parent).with_suffix(".md")
        # full_doc_path = Path("docs/reference/") / doc_path
        parts = list(module_path.parts)
        current = reference_nav
        for part in parts[:-1]:
            sub = next((item[part] for item in current if part in item), None)
            if sub is None:
                current.append({part: []})
                sub = current[-1][part]
            current = sub
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            current.append({"index.md": str(doc_path)})
        elif parts[-1] == "__main__":
            continue
        else:
            current.append({parts[-1]: str(doc_path)})
        ident = ".".join(parts)
        os.makedirs(doc_path.parent, exist_ok=True)
        VIRTUAL_FILES[str(doc_path)] = REFERENCE_TEMPLATE.format(ident=ident)

    for item in config["nav"]:
        if not isinstance(item, dict):
            continue
        key = next(iter(item.keys()))
        if not isinstance(item[key], str):
            continue
        if item[key].strip("/") == "reference":
            item[key] = reference_nav

    VIRTUAL_FILES["contributing.md"] = Path("contributing.md").read_text()
    VIRTUAL_FILES["changelog.md"] = Path("changelog.md").read_text()

    return mkdocs.structure.files.Files(
        [file for file in files if not exclude_file(file.src_path)]
        + [
            mkdocs.structure.files.File(
                file,
                config["docs_dir"],
                config["site_dir"],
                config["use_directory_urls"],
            )
            for file in VIRTUAL_FILES
        ]
    )


def on_nav(nav, config, files):
    def rec(node):
        if isinstance(node, list):
            return [rec(item) for item in node]
        if node.is_section and node.title == "Code Reference":
            return
        if isinstance(node, mkdocs.structure.nav.Navigation):
            return rec(node.items)
        if isinstance(node, mkdocs.structure.nav.Section):
            if (
                len(node.children)
                and node.children[0].is_page
                and node.children[0].is_index
            ):
                first = node.children[0]
                link = mkdocs.structure.nav.Link(
                    title=first.title,
                    url=first.url,
                )
                link.is_index = True
                first.title = "Overview"
                node.children.insert(0, link)
            return rec(node.children)

    rec(nav.items)


def on_page_read_source(page, config):
    if page.file.src_path in VIRTUAL_FILES:
        return VIRTUAL_FILES[page.file.src_path]
    return None


# Get current git commit
GIT_COMMIT = os.popen("git rev-parse --short HEAD").read().strip()


@mkdocs.plugins.event_priority(-2000)
def on_post_page(
    output: str,
    page: mkdocs.structure.pages.Page,
    config: mkdocs.config.Config,
):
    """
    Add github links to the html output
    """
    # Find all the headings (h1, h2, ...) whose id starts with "edsnlp"
    soup = BeautifulSoup(output, "html.parser")
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        ref = heading.get("id", "")
        if ref.startswith("edsnlp.") and "--" not in ref:
            code = "import edsnlp; " + ref
            interpreter = jedi.Interpreter(code, namespaces=[{}])
            goto = interpreter.infer(1, len(code))
            try:
                file = goto[0].module_path.relative_to(Path.cwd())
            except Exception:
                goto = []
            if not goto:
                continue
            line = goto[0].line
            # Add a "[source]" span with a link to the source code in a new tab
            url = f"https://github.com/aphp/edsnlp/blob/{GIT_COMMIT}/{file}#L{line}"
            heading.append(
                BeautifulSoup(
                    f'<span class="sourced-heading-spacer"></span>'
                    f'<a href="{url}" target="_blank">[source]</a>',
                    features="html.parser",
                )
            )
            # add "sourced-heading" to heading class
            heading["class"] = heading.get("class", []) + ["sourced-heading"]
    return str(soup)
