import os
import re
from pathlib import Path

import mkdocs.config
import mkdocs.plugins
import mkdocs.structure
import mkdocs.structure.files
import mkdocs.structure.nav
import mkdocs.structure.pages
from mkdocs.config.defaults import MkDocsConfig

from docs.scripts.autorefs.plugin import AutorefsPlugin


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


@mkdocs.plugins.event_priority(1000)
def on_config(config: MkDocsConfig):
    for event_name, events in config.plugins.events.items():
        for event in list(events):
            if "autorefs" in str(event):
                print("REMOVING EVENT", event_name, event)
                events.remove(event)
    old_plugin = config["plugins"]["autorefs"]
    plugin_config = dict(old_plugin.config)
    plugin = AutorefsPlugin()
    config.plugins["autorefs"] = plugin
    config["plugins"]["autorefs"] = plugin
    plugin.load_config(plugin_config)


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
                and not node.children[0].is_index
            ):
                first = node.children[0]
                link = mkdocs.structure.nav.Link(
                    title=first.title,
                    url=first.url,
                )
                link.is_index = True
                node.children.append(link)
            return rec(node.children)

    rec(nav.items)


def on_page_read_source(page, config):
    if page.file.src_path in VIRTUAL_FILES:
        return VIRTUAL_FILES[page.file.src_path]
    return None


HREF_REGEX = r'href=(?:"([^"]*)"|\'([^\']*)|[ ]*([^ =>]*)(?![a-z]+=))'


@mkdocs.plugins.event_priority(-1000)
def on_post_page(
    output: str,
    page: mkdocs.structure.pages.Page,
    config: mkdocs.config.Config,
):
    """
    Replace absolute paths with path relative to the rendered page
    This must be performed after all other plugins have run.

    Parameters
    ----------
    output
    page
    config

    Returns
    -------

    """

    def replace_link(match):
        relative_url = url = match.group(1) or match.group(2) or match.group(3)
        page_url = os.path.join("/", page.file.url)
        if url.startswith("/"):
            relative_url = os.path.relpath(url, page_url)
        return f'href="{relative_url}"'

    # Replace absolute paths with path relative to the rendered page
    return re.sub(HREF_REGEX, replace_link, output)
