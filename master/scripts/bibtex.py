# Based on https://github.com/darwindarak/mdx_bib
import re
import string
from collections import Counter, OrderedDict
from typing import Tuple
from xml.etree import ElementTree as etree
from xml.etree.ElementTree import tostring as etree_to_string

from markdown.extensions import Extension
from markdown.inlinepatterns import Pattern
from markdown.preprocessors import Preprocessor
from mkdocs.config.config_options import Type as MkType
from mkdocs.plugins import BasePlugin
from pybtex.database.input import bibtex
from pybtex.exceptions import PybtexError

BRACKET_RE = re.compile(r"\[([^\[]+)\]")
CITE_RE = re.compile(r"@([\w_:-]+)")
DEF_RE = re.compile(r"\A {0,3}\[@([\w_:-]+)\]:\s*(.*)")
INDENT_RE = re.compile(r"\A\t| {4}(.*)")

CITATION_RE = r"(\[@(?:[\w_:-]+)(?: *, *@(?:[\w_:-]+))*\])"


class Bibliography(object):
    """Keep track of document references and citations for exporting"""

    def __init__(self, extension, plugin, bibtex_file, order):
        self.extension = extension
        self.order = order
        self.plugin = plugin

        self.citations = OrderedDict()
        self.references = dict()

        if bibtex_file:
            try:
                parser = bibtex.Parser()
                self.bibsource = parser.parse_file(bibtex_file).entries
                self.labels = {
                    id: self.formatCitation(self.bibsource[id])
                    for id in self.bibsource.keys()
                }
                for value, occurrences in Counter(self.labels.values()).items():
                    if occurrences > 1:
                        for xkey, xvalue in self.labels.items():
                            i = 0
                            if xvalue == value:
                                self.labels[
                                    xkey
                                ] = f"{xvalue}{string.ascii_lowercase[i]}"
                                i += 1

            except PybtexError:
                print("Error loading bibtex file")
                self.bibsource = dict()
                self.labels = {}
        else:
            self.bibsource = dict()

    def addCitation(self, citekey):
        self.citations[citekey] = self.citations.get(citekey, 0) + 1

    def setReference(self, citekey, reference):
        self.references[citekey] = reference

    def citationID(self, citekey):
        return "cite-" + citekey

    def referenceID(self, citekey):
        return "ref-" + citekey

    def formatAuthor(self, author):
        out = (
            author.last_names[0]
            + ((" " + author.first_names[0][0]) if author.first_names else "")
            + "."
        )
        if author.middle_names:
            out += f"{author.middle_names[0][0]}."
        return out.replace("{", "").replace("}", "")

    def formatAuthorSurname(self, author):
        out = author.last_names[0]
        return out.replace("{", "").replace("}", "")

    def formatReference(self, ref):
        author_list = list(map(self.formatAuthor, ref.persons["author"]))

        if len(author_list) == 1:
            authors = author_list[0]
        else:
            authors = ", ".join(author_list[:-1])
            authors += f" and {author_list[-1]}"

        # Harvard style
        # Surname, Initial, ... and Last_Surname,
        # Initial, Year. Title. Journal, Volume(Issue), pages. doi.

        title = ref.fields["title"].replace("{", "").replace("}", "")
        journal = ref.fields.get("journal", "")
        volume = ref.fields.get("volume", "")
        issue = ref.fields.get("issue", "")
        year = ref.fields.get("year")
        pages = ref.fields.get("pages")
        doi = ref.fields.get("doi")

        ref_id = self.referenceID(ref.key)
        reference = f"<p id={repr(ref_id)}>{authors}, {year}. {title}."
        if journal:
            reference += f" <i>{journal}</i>."
            if volume:
                reference += f" <i>{volume}</i>"
            if issue:
                reference += f"({issue})"
            if pages:
                reference += f", pp.{pages}"
            reference += "."
        if doi:
            reference += (
                f' <a href="https://dx.doi.org/{doi}" target="_blank">{doi}</a>'
            )
        reference += "</p>"

        return etree.fromstring(reference)

    def formatCitation(self, ref):
        author_list = list(map(self.formatAuthorSurname, ref.persons["author"]))
        year = ref.fields.get("year")

        if len(author_list) == 1:
            citation = f"{author_list[0]}"
        elif len(author_list) == 2:
            citation = f"{author_list[0]} and {author_list[1]}"
        else:
            citation = f"{author_list[0]} et al."

        citation += f", {year}"

        return citation

    def make_bibliography(self):
        if self.order == "alphabetical":
            raise (NotImplementedError)

        div = etree.Element("div")
        div.set("class", "footnote")
        div.append(etree.Element("hr"))
        ol = etree.SubElement(div, "ol")

        if not self.citations:
            return div

        # table = etree.SubElement(div, "table")
        # table.set("class", "references")
        # tbody = etree.SubElement(table, "tbody")
        etree.SubElement(div, "div")
        for id in self.citations:
            li = etree.SubElement(ol, "li")
            li.set("id", self.referenceID(id))
            # ref_id = etree.SubElement(li, "td")
            ref_txt = etree.SubElement(li, "p")
            if id in self.references:
                self.extension.parser.parseChunk(ref_txt, self.references[id])
            elif id in self.bibsource:
                ref_txt.append(self.formatReference(self.bibsource[id]))
            else:
                ref_txt.text = "Missing citation for {}".format(id)

        return div

    def clear_citations(self):
        self.citations = OrderedDict()


class CitationsPreprocessor(Preprocessor):
    """Gather reference definitions and citation keys"""

    def __init__(self, bibliography):
        self.bib = bibliography

    def subsequentIndents(self, lines, i):
        """Concatenate consecutive indented lines"""
        linesOut = []
        while i < len(lines):
            m = INDENT_RE.match(lines[i])
            if m:
                linesOut.append(m.group(1))
                i += 1
            else:
                break
        return " ".join(linesOut), i

    def run(self, lines):
        linesOut = []
        i = 0

        while i < len(lines):
            # Check to see if the line starts a reference definition
            m = DEF_RE.match(lines[i])
            if m:
                key = m.group(1)
                reference = m.group(2)
                indents, i = self.subsequentIndents(lines, i + 1)
                reference += " " + indents

                self.bib.setReference(key, reference)
                continue

            # Look for all @citekey patterns inside hard brackets
            for bracket in BRACKET_RE.findall(lines[i]):
                for c in CITE_RE.findall(bracket):
                    self.bib.addCitation(c)
            linesOut.append(lines[i])
            i += 1

        return linesOut


class CitationsPattern(Pattern):
    """Handles converting citations keys into links"""

    def __init__(self, pattern, bibliography):
        super(CitationsPattern, self).__init__(pattern)
        self.bib = bibliography

    def handleMatch(self, m):
        span = etree.Element("span")
        for cite_match in CITE_RE.finditer(m.group(2)):
            id = cite_match.group(1)
            if id in self.bib.bibsource:
                a = etree.Element("a")
                a.set("id", self.bib.citationID(id))
                a.set("href", "./#" + self.bib.referenceID(id))
                a.set("class", "citation")
                a.text = self.bib.labels[id]
                span.append(a)
            else:
                continue
        if len(span) == 0:
            return None
        return span


context_citations = None


class CitationsExtension(Extension):
    def __init__(self):
        super(CitationsExtension, self).__init__()
        self.bib = None

    def extendMarkdown(self, md):
        md.registerExtension(self)
        self.parser = md.parser
        self.md = md

        md.preprocessors.register(CitationsPreprocessor(self.bib), "mdx_bib", 15)
        md.inlinePatterns.register(
            CitationsPattern(CITATION_RE, self.bib), "mdx_bib", 175
        )


def makeExtension(*args, **kwargs):
    return CitationsExtension(*args, **kwargs)


class BibTexPlugin(BasePlugin):
    config_scheme: Tuple[Tuple[str, MkType]] = (
        ("bibtex_file", MkType(str)),  # type: ignore[assignment]
        ("order", MkType(str, default="unsorted")),  # type: ignore[assignment]
    )

    def __init__(self):
        self.citations = None

    def on_config(self, config, **kwargs):
        extension = CitationsExtension()
        self.bib = Bibliography(
            extension,
            self,
            self.config["bibtex_file"],
            self.config["order"],
        )
        extension.bib = self.bib
        config["markdown_extensions"].append(extension)

    def on_page_content(self, html, page, config, files):
        html += "\n" + etree_to_string(self.bib.make_bibliography()).decode()
        self.bib.clear_citations()
        return html
