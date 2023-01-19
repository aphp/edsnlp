from io import StringIO
from typing import Dict, Optional, Union

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.pipelines.misc.tables import patterns
from edsnlp.utils.filter import get_spans


class Tables(GenericMatcher):
    """Pipeline to identify the Tables.

    It adds the key `tables` to doc.spans.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    tables_pattern : Optional[Dict[str, str]]
        The regex pattern to identify tables.
        The key of dictionary should be `tables`
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with
        the key 'term_attr'. We can also add a key for each regex.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        nlp: Language,
        tables_pattern: Optional[Dict[str, str]],
        sep_pattern: Optional[str],
        attr: Union[Dict[str, str], str],
        ignore_excluded: bool,
    ):

        if tables_pattern is None:
            self.tables_pattern = patterns.regex
        else:
            self.tables_pattern = tables_pattern

        if sep_pattern is None:
            self.sep = patterns.sep
        else:
            self.sep = sep_pattern

        super().__init__(
            nlp,
            terms=None,
            regex=self.tables_pattern,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        if not Span.has_extension("to_pd_table"):
            Span.set_extension("to_pd_table", getter=self.to_pd_table)

        self.set_extensions()

    def __call__(self, doc: Doc) -> Doc:
        """Find spans that contain tables

        Parameters
        ----------
        doc : Doc

        Returns
        -------
        Doc
        """
        matches = self.process(doc)
        tables = get_spans(matches, "tables")
        # parsed = self.parse(tables=tables)

        doc.spans["tables"] = tables

        return doc

    def to_pd_table(self, span) -> pd.DataFrame:
        table_str_io = StringIO(span.text)
        parsed = pd.read_csv(
            table_str_io,
            sep=self.sep,
            engine="python",
            header=None,
            error_bad_lines=False,
        )
        return parsed
