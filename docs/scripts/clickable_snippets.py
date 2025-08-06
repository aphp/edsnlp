# Based on https://github.com/darwindarak/mdx_bib
import os
from bisect import bisect_right
from collections import defaultdict
from typing import Tuple

import jedi
import mkdocs.structure.pages
import parso
import regex
from mkdocs.config.config_options import Type as MkType
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin

from docs.scripts.autorefs.plugin import AutorefsPlugin

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points


from bs4 import BeautifulSoup

# Used to match href in HTML to replace with a relative path
HREF_REGEX = (
    r"(?<=<\s*(?:a[^>]*href|img[^>]*src)=)"
    r'(?:"([^"]*)"|\'([^\']*)|[ ]*([^ =>]*)(?![a-z]+=))'
)
# Maybe find something less specific ?
PIPE_REGEX = r"(?<![a-zA-Z0-9._-])eds[.]([a-zA-Z0-9._-]*)(?![a-zA-Z0-9._-])"

HTML_PIPE_REGEX = r"""(?x)
(?<![a-zA-Z0-9._-])
<span[^>]*>eds<\/span>
<span[^>]*>[.]<\/span>
<span[^>]*>([a-zA-Z0-9._-]*)<\/span>
(?![a-zA-Z0-9._-])
"""

REGISTRY_REGEX = r"""(?x)
(?<![a-zA-Z0-9._-])
<span[^>]*>(?:"|&\#39;|&quot;)@([a-zA-Z0-9._-]*)(?:"|&\#39;|&quot;)<\/span>\s*
<span[^>]*>:<\/span>\s*
<span[^>]*>\s*<\/span>\s*
<span[^>]*>(?:"|&\#39;|&quot;)?([a-zA-Z0-9._-]*)(?:"|&\#39;|&quot;)?<\/span>
(?![a-zA-Z0-9._-])
"""

CITATION_RE = r"(\[@(?:[\w_:-]+)(?: *, *@(?:[\w_:-]+))*\])"


class ClickableSnippetsPlugin(BasePlugin):
    config_scheme: Tuple[Tuple[str, MkType]] = ()

    @mkdocs.plugins.event_priority(1000)
    def on_config(self, config: MkDocsConfig):
        for event_name, events in config.plugins.events.items():
            for event in list(events):
                if "autorefs" in str(event):
                    events.remove(event)
        old_plugin = config["plugins"]["autorefs"]
        plugin_config = dict(old_plugin.config)
        plugin = AutorefsPlugin()
        config.plugins["autorefs"] = plugin
        config["plugins"]["autorefs"] = plugin
        plugin.load_config(plugin_config)

    @classmethod
    def get_ep_namespace(cls, ep, namespace=None):
        if hasattr(ep, "select"):
            return ep.select(group=namespace) if namespace else list(ep._all)
        else:  # dict
            return (
                ep.get(namespace, [])
                if namespace
                else (x for g in ep.values() for x in g)
            )

    @mkdocs.plugins.event_priority(-1000)
    def on_post_page(
        self,
        output: str,
        page: mkdocs.structure.pages.Page,
        config: mkdocs.config.Config,
    ):
        """
        1. Replace absolute paths with path relative to the rendered page
           This must be performed after all other plugins have run.
        2. Replace component names with links to the component reference

        Parameters
        ----------
        output
        page
        config

        Returns
        -------

        """

        autorefs: AutorefsPlugin = config["plugins"]["autorefs"]
        ep = entry_points()
        page_url = os.path.join("/", page.file.url)
        factories_entry_points = {
            ep.name: ep.value
            for ep in (
                *self.get_ep_namespace(ep, "spacy_factories"),
                *self.get_ep_namespace(ep, "edsnlp_factories"),
                *self.get_ep_namespace(ep, "spacy_scorers"),
            )
        }
        all_entry_points = defaultdict(dict)
        for ep in self.get_ep_namespace(ep):
            if ep.group.startswith("edsnlp_") or ep.group.startswith("spacy_"):
                group = ep.group.split("_", 1)[1]
                all_entry_points[group][ep.name] = ep.value

        # This method is meant for replacing any component that
        # appears in a "eds.component" format, no matter if it is
        # preceded by a "@factory" or not.
        def replace_factory_component(match):
            full_match = match.group(0)
            name = "eds." + match.group(1)
            ep = factories_entry_points.get(name)
            preceding = output[match.start(0) - 50 : match.start(0)]
            if ep is not None and "DEFAULT:" not in preceding:
                try:
                    url = autorefs.get_item_url(ep.replace(":", "."))
                except KeyError:
                    pass
                else:
                    return f"<a href={url}>{name}</a>"
            return full_match

        # This method is meant for replacing any component that
        # appears in a "@registry": "component" format
        def replace_any_registry_component(match):
            full_match = match.group(0)
            group = match.group(1)
            name = match.group(2)
            ep = all_entry_points[group].get(name)
            preceding = output[match.start(0) - 50 : match.start(0)]
            if ep is not None and "DEFAULT:" not in preceding:
                try:
                    url = autorefs.get_item_url(ep.replace(":", "."))
                except KeyError:
                    pass
                else:
                    repl = f'<a href={url} class="discrete-link">{name}</a>'
                    before = full_match[: match.start(2) - match.start(0)]
                    after = full_match[match.end(2) - match.start(0) :]
                    return before + repl + after
            return full_match

        def replace_link(match):
            relative_url = url = match.group(1) or match.group(2) or match.group(3)
            if url.startswith("/"):
                relative_url = os.path.relpath(url, page_url)
            return f'"{relative_url}"'

        output = regex.sub(PIPE_REGEX, replace_factory_component, output)
        output = regex.sub(HTML_PIPE_REGEX, replace_factory_component, output)
        output = regex.sub(REGISTRY_REGEX, replace_any_registry_component, output)

        all_snippets = ""
        all_offsets = []
        all_nodes = []

        soups = []

        # Replace absolute paths with path relative to the rendered page
        for match in regex.finditer("<code>.*?</code>", output, flags=regex.DOTALL):
            node = match.group(0)
            if "\n" in node:
                soup, snippet, python_offsets, html_nodes = self.convert_html_to_code(
                    node
                )
                size = len(all_snippets)
                all_snippets += snippet + "\n"
                all_offsets.extend([size + i for i in python_offsets])
                all_nodes.extend(html_nodes)
                soups.append((soup, match.start(0), match.end(0)))

        interpreter = jedi.Interpreter(all_snippets, [{}])
        line_lengths = [0]
        for line in all_snippets.split("\n"):
            line_lengths.append(len(line) + line_lengths[-1] + 1)
        line_lengths[-1] -= 1

        for name in self.iter_names(interpreter._module_node):
            try:
                line, col = name.start_pos
                offset = line_lengths[line - 1] + col
                node_idx = bisect_right(all_offsets, offset) - 1

                node = all_nodes[node_idx]
                gotos = interpreter.goto(line, col, follow_imports=True)
                gotos = [
                    goto
                    for goto in gotos
                    if (
                        goto
                        and goto.full_name
                        and goto.full_name.startswith("edsnlp")
                        and goto.type != "module"
                    )
                ]
                goto = gotos[0] if gotos else None
                if goto:
                    url = autorefs.get_item_url(goto.full_name)
                    # Check if node has no link in its upstream ancestors
                    if not node.find_parents("a"):
                        node.replace_with(
                            BeautifulSoup(
                                f'<a class="discrete-link" href="{url}">{node}</a>',
                                "html5lib",
                            )
                        )
            except Exception:
                pass

        # Re-insert soups into the output
        for soup, start, end in reversed(soups):
            output = output[:start] + str(soup.find("code")) + output[end:]

        output = regex.sub(HREF_REGEX, replace_link, output)

        return output

    @classmethod
    def iter_names(cls, root):
        if isinstance(root, parso.python.tree.Name):
            yield root
        for child in getattr(root, "children", ()):
            yield from cls.iter_names(child)

    @classmethod
    def convert_html_to_code(
        cls, html_content: str
    ) -> Tuple[BeautifulSoup, str, list, list]:
        pre_html_content = "<pre>" + html_content + "</pre>"
        soup = list(BeautifulSoup(pre_html_content, "html5lib").children)[0]
        code_element = soup.find("code")

        line_lengths = [0]
        for line in pre_html_content.split("\n"):
            line_lengths.append(len(line) + line_lengths[-1] + 1)
        line_lengths[-1] -= 1

        python_code = ""
        code_offsets = []
        html_nodes = []
        code_offset = 0

        def extract_text_with_offsets(el):
            nonlocal python_code, code_offset
            for content in el.contents:
                # check not class md-annotation
                # Recursively process child elements
                if isinstance(content, str):
                    python_code += content
                    code_offsets.append(code_offset)
                    code_offset += len(content)
                    html_nodes.append(content)
                    continue
                if "md-annotation" not in content.get("class", ""):
                    extract_text_with_offsets(content)

        extract_text_with_offsets(code_element)

        return soup, python_code, code_offsets, html_nodes
