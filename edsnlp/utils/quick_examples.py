import bisect
from typing import List, Union

from rich.console import Console
from rich.table import Table
from spacy.language import Language
from spacy.tokens import Doc, Token

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.qualifiers.base import get_qualifier_extensions
from edsnlp.utils.extensions import rgetattr


class QuickExamples:
    def __init__(self, nlp: Language, extensions: List[str] = []):
        self.nlp = nlp
        self.qualifiers = get_qualifier_extensions(nlp)
        self.extensions = extensions

    def __call__(self, object: Union[str, Doc]):
        if isinstance(object, str):
            self.txt = object
            self.doc = self.nlp(object)
        elif isinstance(object, Doc):
            self.txt = object.text
            self.doc = object
        self.get_ents()
        self.get_ents_interval()
        self.get_text()
        self.display()

    def get_ents(self):

        all_spans = {k: list(s) for k, s in self.doc.spans.items() if s}
        all_spans["ents"] = list(self.doc.ents).copy()

        ents = []

        for key, spans in all_spans.items():
            for span in spans:
                if span in all_spans["ents"]:
                    all_spans["ents"].remove(span)
                start, end = span.start, span.end
                text = get_text(span, attr="TEXT", ignore_excluded=False)
                ent = dict(
                    key=key,
                    start=start,
                    end=end,
                    text=text,
                )
                for name, extension in self.qualifiers.items():
                    ent[name] = rgetattr(span, extension)
                for extension in self.extensions:
                    ent[extension] = rgetattr(span, extension)
                ents.append(ent)

        self.ents = ents

    def get_ents_interval(self):
        """
        From the list of all entities, removes overlapping spans
        """

        intervals = []
        for ent in self.ents:
            interval = (ent["start"], ent["end"])
            istart, iend = interval

            i = bisect.bisect_right(intervals, (iend, len(self.doc) + 1))

            for idx, (start, end) in enumerate(intervals[:i]):
                if end > istart:
                    interval = (start, iend)
                    del intervals[idx]
                    break

            bisect.insort(intervals, interval)

        self.intervals = intervals

    def is_ent(self, tok: Token) -> bool:
        """
        Check if the provided Token is part of an entity

        Parameters
        ----------
        tok : Token
            A spaCy Token

        Returns
        -------
        bool
            True if `tok` is part of an entity
        """
        for interval in self.intervals:
            if (tok.i >= interval[0]) and (tok.i < interval[1]):
                return True
        return False

    def get_text(self) -> None:
        """
        Adds bold tags to `self.text`
        """
        text = []
        for tok in self.doc:
            raw_tok_text = tok.text + tok.whitespace_
            tok_text = (
                f"[bold]{raw_tok_text}[not bold]" if self.is_ent(tok) else raw_tok_text
            )
            text.append(tok_text)
        self.text = "".join(text)

    def display(self) -> None:
        """
        Displays the text and a table of entities
        """
        console = Console()

        table = Table(title=self.text + "\n")

        table.add_column("Entity")
        table.add_column("Source")

        for q in self.qualifiers:
            table.add_column(q)
        for e in self.extensions:
            table.add_column(e)

        for ent in self.ents:
            table.add_row(
                ent["text"],
                ent["key"],
                *(
                    "[green]" + str(ent[q]) if ent[q] else "[red]" + str(ent[q])
                    for q in self.qualifiers
                ),
                *(str(ent[extension]) for extension in self.extensions),
            )

        console.print(table)
