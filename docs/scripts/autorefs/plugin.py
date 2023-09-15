# ruff: noqa: E501
"""
# -----------
VENDORED https://github.com/mkdocstrings/autorefs/blob/e19b9fa47dac136a529c2be0d7969106ca5d5106/src/mkdocs_autorefs/
Waiting for the following PR to be merged: https://github.com/mkdocstrings/autorefs/pull/25
# -----------

This module contains the "mkdocs-autorefs" plugin.

After each page is processed by the Markdown converter, this plugin stores absolute URLs of every HTML anchors
it finds to later be able to fix unresolved references.
It stores them during the [`on_page_content` event hook](https://www.mkdocs.org/user-guide/plugins/#on_page_content).

Just before writing the final HTML to the disc, during the
[`on_post_page` event hook](https://www.mkdocs.org/user-guide/plugins/#on_post_page),
this plugin searches for references of the form `[identifier][]` or `[title][identifier]` that were not resolved,
and fixes them using the previously stored identifier-URL mapping.
"""
import contextlib
import functools
import logging
import os
import re
from html import escape, unescape
from typing import Any, Callable, Dict, List, Match, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element

from markdown import Markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import REFERENCE_RE, ReferenceInlineProcessor
from markdown.util import INLINE_PLACEHOLDER_RE
from mkdocs.config import Config
from mkdocs.config import config_options as c
from mkdocs.plugins import BasePlugin
from mkdocs.structure.pages import Page
from mkdocs.structure.toc import AnchorLink
from mkdocs.utils import warning_filter

AUTO_REF_RE = re.compile(
    r"<span data-(?P<kind>autorefs-identifier|autorefs-optional|autorefs-optional-hover)="
    r'("?)(?P<identifier>[^"<>]*)\2>(?P<title>.*?)</span>'
)
"""A regular expression to match mkdocs-autorefs' special reference markers
in the [`on_post_page` hook][mkdocs_autorefs.plugin.AutorefsPlugin.on_post_page].
"""

EvalIDType = Tuple[Any, Any, Any]


class AutoRefInlineProcessor(ReferenceInlineProcessor):
    """A Markdown extension."""

    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(REFERENCE_RE, *args, **kwargs)

    # Code based on
    # https://github.com/Python-Markdown/markdown/blob/8e7528fa5c98bf4652deb13206d6e6241d61630b/markdown/inlinepatterns.py#L780

    def handleMatch(self, m, data) -> Union[Element, EvalIDType]:  # type: ignore[override]  # noqa: N802,WPS111
        """Handle an element that matched.

        Arguments:
            m: The match object.
            data: The matched data.

        Returns:
            A new element or a tuple.
        """
        text, index, handled = self.getText(data, m.end(0))
        if not handled:
            return None, None, None

        identifier, end, handled = self.evalId(data, index, text)
        if not handled:
            return None, None, None

        if re.search(r"[/ \x00-\x1f]", identifier):
            # Do nothing if the matched reference contains:
            # - a space, slash or control character (considered unintended);
            # - specifically \x01 is used by Python-Markdown HTML stash when there's inline formatting,
            #   but references with Markdown formatting are not possible anyway.
            return None, m.start(0), end

        return self.makeTag(identifier, text), m.start(0), end

    def evalId(
        self, data: str, index: int, text: str
    ) -> EvalIDType:  # noqa: N802 (parent's casing)
        """Evaluate the id portion of `[ref][id]`.

        If `[ref][]` use `[ref]`.

        Arguments:
            data: The data to evaluate.
            index: The starting position.
            text: The text to use when no identifier.

        Returns:
            A tuple containing the identifier, its end position, and whether it matched.
        """
        m = self.RE_LINK.match(data, pos=index)  # noqa: WPS111
        if not m:
            return None, index, False

        identifier = m.group(1)
        if not identifier:
            identifier = text
            # Allow the entire content to be one placeholder, with the intent of catching things like [`Foo`][].
            # It doesn't catch [*Foo*][] though, just due to the priority order.
            # https://github.com/Python-Markdown/markdown/blob/1858c1b601ead62ed49646ae0d99298f41b1a271/markdown/inlinepatterns.py#L78
            if INLINE_PLACEHOLDER_RE.fullmatch(identifier):
                identifier = self.unescape(identifier)

        end = m.end(0)
        return identifier, end, True

    def makeTag(self, identifier: str, text: str) -> Element:  # type: ignore[override]  # noqa: N802,W0221
        """Create a tag that can be matched by `AUTO_REF_RE`.

        Arguments:
            identifier: The identifier to use in the HTML property.
            text: The text to use in the HTML tag.

        Returns:
            A new element.
        """
        el = Element("span")
        el.set("data-autorefs-identifier", identifier)
        el.text = text
        return el


def relative_url(url_a: str, url_b: str) -> str:
    """Compute the relative path from URL A to URL B.

    Arguments:
        url_a: URL A.
        url_b: URL B.

    Returns:
        The relative URL to go from A to B.
    """
    parts_a = url_a.split("/")
    url_b, anchor = url_b.split("#", 1)
    parts_b = url_b.split("/")

    # remove common left parts
    while parts_a and parts_b and parts_a[0] == parts_b[0]:
        parts_a.pop(0)
        parts_b.pop(0)

    # go up as many times as remaining a parts' depth
    levels = len(parts_a) - 1
    parts_relative = [".."] * levels + parts_b  # noqa: WPS435
    relative = "/".join(parts_relative)
    return f"{relative}#{anchor}"


def fix_ref(
    url_mapper: Callable[[str], str], unmapped: List[str]
) -> Callable:  # noqa: WPS212,WPS231
    """Return a `repl` function for [`re.sub`](https://docs.python.org/3/library/re.html#re.sub).

    In our context, we match Markdown references and replace them with HTML links.

    When the matched reference's identifier was not mapped to an URL, we append the identifier to the outer
    `unmapped` list. It generally means the user is trying to cross-reference an object that was not collected
    and rendered, making it impossible to link to it. We catch this exception in the caller to issue a warning.

    Arguments:
        url_mapper: A callable that gets an object's site URL by its identifier,
            such as [mkdocs_autorefs.plugin.AutorefsPlugin.get_item_url][].
        unmapped: A list to store unmapped identifiers.

    Returns:
        The actual function accepting a [`Match` object](https://docs.python.org/3/library/re.html#match-objects)
        and returning the replacement strings.
    """

    def inner(match: Match):  # noqa: WPS212,WPS430
        identifier = match["identifier"]
        title = match["title"]
        kind = match["kind"]

        try:
            url = url_mapper(unescape(identifier))
        except KeyError:
            if kind == "autorefs-optional":
                return title
            elif kind == "autorefs-optional-hover":
                return f'<span title="{identifier}">{title}</span>'
            unmapped.append(identifier)
            if title == identifier:
                return f"[{identifier}][]"
            return f"[{title}][{identifier}]"

        parsed = urlsplit(url)
        external = parsed.scheme or parsed.netloc
        classes = ["autorefs", "autorefs-external" if external else "autorefs-internal"]
        class_attr = " ".join(classes)
        if kind == "autorefs-optional-hover":
            return f'<a class="{class_attr}" title="{identifier}" href="{escape(url)}">{title}</a>'
        return f'<a class="{class_attr}" href="{escape(url)}">{title}</a>'

    return inner


def fix_refs(html: str, url_mapper: Callable[[str], str]) -> Tuple[str, List[str]]:
    """Fix all references in the given HTML text.

    Arguments:
        html: The text to fix.
        url_mapper: A callable that gets an object's site URL by its identifier,
            such as [mkdocs_autorefs.plugin.AutorefsPlugin.get_item_url][].

    Returns:
        The fixed HTML.
    """
    unmapped = []  # type: ignore
    html = AUTO_REF_RE.sub(fix_ref(url_mapper, unmapped), html)
    return html, unmapped


class AutorefsExtension(Extension):
    """Extension that inserts auto-references in Markdown."""

    def extendMarkdown(
        self, md: Markdown
    ) -> None:  # noqa: N802 (casing: parent method's name)
        """Register the extension.

        Add an instance of our [`AutoRefInlineProcessor`][mkdocs_autorefs.references.AutoRefInlineProcessor] to the Markdown parser.

        Arguments:
            md: A `markdown.Markdown` instance.
        """
        md.inlinePatterns.register(
            AutoRefInlineProcessor(md),
            "mkdocs-autorefs",
            priority=168,  # noqa: WPS432  # Right after markdown.inlinepatterns.ReferenceInlineProcessor
        )


log = logging.getLogger(f"mkdocs.plugins.{__name__}")
log.addFilter(warning_filter)


class AutorefsPlugin(BasePlugin):
    """An `mkdocs` plugin.

    This plugin defines the following event hooks:

    - `on_config`
    - `on_page_content`
    - `on_post_page`

    Check the [Developing Plugins](https://www.mkdocs.org/user-guide/plugins/#developing-plugins) page of `mkdocs`
    for more information about its plugin system.
    """

    scan_toc: bool = True
    current_page: Optional[str] = None
    config_scheme = (("priority", c.ListOfItems(c.Type(str), default=[])),)

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()
        self._url_map: Dict[str, str] = {}
        self._abs_url_map: Dict[str, str] = {}
        self.get_fallback_anchor: Optional[
            Callable[[str], Optional[str]]
        ] = None  # noqa: WPS234
        self._priority_patterns = None

    @property
    def priority_patterns(self):
        if self._priority_patterns is None:
            self._priority_patterns = [
                os.path.join("/", pat) for pat in self.config.get("priority")
            ]
        return self._priority_patterns

    def register_anchor(self, url: str, identifier: str):
        """Register that an anchor corresponding to an identifier was encountered when rendering the page.

        Arguments:
            url: The relative URL of the current page. Examples: `'foo/bar/'`, `'foo/index.html'`
            identifier: The HTML anchor (without '#') as a string.
        """

        new_url = os.path.join("/", f"{url}#{identifier}")
        old_url = os.path.join("/", self._url_map.get(identifier, "")).split("#")[0]

        if identifier in self._url_map and not old_url == new_url:
            rev_patterns = list(enumerate(self.priority_patterns))[::-1]
            old_priority_idx = next(
                (i for i, pat in rev_patterns if re.match(pat, old_url)),
                len(rev_patterns),
            )
            new_priority_idx = next(
                (i for i, pat in rev_patterns if re.match(pat, new_url)),
                len(rev_patterns),
            )
            if new_priority_idx >= old_priority_idx:
                return
            if "reference" not in new_url:
                raise Exception("URL WTF", new_url)

        self._url_map[identifier] = new_url

    def register_url(self, identifier: str, url: str):
        """Register that the identifier should be turned into a link to this URL.

        Arguments:
            identifier: The new identifier.
            url: The absolute URL (including anchor, if needed) where this item can be found.
        """
        self._abs_url_map[identifier] = url

    def _get_item_url(  # noqa: WPS234
        self,
        identifier: str,
        fallback: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> str:
        try:
            return self._url_map[identifier]
        except KeyError:
            if identifier in self._abs_url_map:
                return self._abs_url_map[identifier]
            if fallback:
                new_identifiers = fallback(identifier)
                for new_identifier in new_identifiers:
                    with contextlib.suppress(KeyError):
                        url = self._get_item_url(new_identifier)
                        self._url_map[identifier] = url
                        return url
            raise

    def get_item_url(  # noqa: WPS234
        self,
        identifier: str,
        from_url: Optional[str] = None,
        fallback: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> str:
        """Return a site-relative URL with anchor to the identifier, if it's present anywhere.

        Arguments:
            identifier: The anchor (without '#').
            from_url: The URL of the base page, from which we link towards the targeted pages.
            fallback: An optional function to suggest alternative anchors to try on failure.

        Returns:
            A site-relative URL.
        """
        return self._get_item_url(identifier, fallback)

    def on_config(
        self, config: Config, **kwargs
    ) -> Config:  # noqa: W0613,R0201 (unused arguments, cannot be static)
        """Instantiate our Markdown extension.

        Hook for the [`on_config` event](https://www.mkdocs.org/user-guide/plugins/#on_config).
        In this hook, we instantiate our [`AutorefsExtension`][mkdocs_autorefs.references.AutorefsExtension]
        and add it to the list of Markdown extensions used by `mkdocs`.

        Arguments:
            config: The MkDocs config object.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            The modified config.
        """
        log.debug(f"{__name__}: Adding AutorefsExtension to the list")
        config["markdown_extensions"].append(AutorefsExtension())
        return config

    def on_page_markdown(
        self, markdown: str, page: Page, **kwargs
    ) -> str:  # noqa: W0613 (unused arguments)
        """Remember which page is the current one.

        Arguments:
            markdown: Input Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            The same Markdown. We only use this hook to map anchors to URLs.
        """
        self.current_page = page.url  # noqa: WPS601
        return markdown

    def on_page_content(
        self, html: str, page: Page, **kwargs
    ) -> str:  # noqa: W0613 (unused arguments)
        """Map anchors to URLs.

        Hook for the [`on_page_content` event](https://www.mkdocs.org/user-guide/plugins/#on_page_content).
        In this hook, we map the IDs of every anchor found in the table of contents to the anchors absolute URLs.
        This mapping will be used later to fix unresolved reference of the form `[title][identifier]` or
        `[identifier][]`.

        Arguments:
            html: HTML converted from Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            The same HTML. We only use this hook to map anchors to URLs.
        """
        if self.scan_toc:
            log.debug(
                f"{__name__}: Mapping identifiers to URLs for page {page.file.src_path}"
            )
            for item in page.toc.items:
                self.map_urls(page, item)
        return html

    def map_urls(self, page: Page, anchor: AnchorLink) -> None:
        """Recurse on every anchor to map its ID to its absolute URL.

        This method populates `self.url_map` by side-effect.

        Arguments:
            base_url: The base URL to use as a prefix for each anchor's relative URL.
            anchor: The anchor to process and to recurse on.
        """
        abs_url = os.path.join("/", page.file.url)
        self.register_anchor(abs_url, anchor.id)
        for child in anchor.children:
            self.map_urls(page, child)

    def on_post_page(
        self, output: str, page: Page, **kwargs
    ) -> str:  # noqa: W0613 (unused arguments)
        """Fix cross-references.

        Hook for the [`on_post_page` event](https://www.mkdocs.org/user-guide/plugins/#on_post_page).
        In this hook, we try to fix unresolved references of the form `[title][identifier]` or `[identifier][]`.
        Doing that allows the user of `autorefs` to cross-reference objects in their documentation strings.
        It uses the native Markdown syntax so it's easy to remember and use.

        We log a warning for each reference that we couldn't map to an URL, but try to be smart and ignore identifiers
        that do not look legitimate (sometimes documentation can contain strings matching
        our [`AUTO_REF_RE`][mkdocs_autorefs.references.AUTO_REF_RE] regular expression that did not intend to reference anything).
        We currently ignore references when their identifier contains a space or a slash.

        Arguments:
            output: HTML converted from Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            Modified HTML.
        """
        log.debug(f"{__name__}: Fixing references in page {page.file.src_path}")

        url_mapper = functools.partial(
            self.get_item_url, from_url=page.url, fallback=self.get_fallback_anchor
        )
        fixed_output, unmapped = fix_refs(output, url_mapper)

        if unmapped and log.isEnabledFor(logging.WARNING):
            for ref in unmapped:
                log.warning(
                    f"{__name__}: {page.file.src_path}: Could not find cross-reference target '[{ref}]'",
                )

        return fixed_output
