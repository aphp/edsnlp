import pickle
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.utils.filter import get_spans

from .endlinesmodel import EndLinesModel
from .functional import build_path


class EndLines(GenericMatcher):
    """
    spaCy Pipeline to detect whether a newline character should
    be considered a space (ie introduced by the PDF).

    The pipeline will add the extension `end_line` to spans
    and tokens. The `end_line` attribute is a boolean or `None`,
    set to `True` if the pipeline predicts that the new line
    is an end line character. Otherwise, it is  set to `False`
    if the new line is classified as a space. If no classification
    has been done over that token, it will remain `None`.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.

    end_lines_model : Optional[Union[str, EndLinesModel]], by default None
        path to trained model. If None, it will use a default model
    """

    def __init__(
        self,
        nlp: Language,
        end_lines_model: Optional[Union[str, EndLinesModel]],
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=None,
            attr="TEXT",
            regex=dict(
                new_line=r"\n+",
            ),
            ignore_excluded=False,
            **kwargs,
        )

        self._read_model(end_lines_model)

    def _read_model(self, end_lines_model: Optional[Union[str, EndLinesModel]]):
        """
        Parameters
        ----------
        end_lines_model : Optional[Union[str, EndLinesModel]]

        Raises
        ------
        TypeError
        """
        if end_lines_model is None:
            path = build_path(__file__, "base_model.pkl")

            with open(path, "rb") as inp:
                self.model = pickle.load(inp)
        elif type(end_lines_model) == str:
            with open(end_lines_model, "rb") as inp:
                self.model = pickle.load(inp)
        elif type(end_lines_model) == EndLinesModel:
            self.model = end_lines_model
        else:
            raise TypeError(
                "type(`end_lines_model`) should be one of {None, str, EndLinesModel}"
            )

    @classmethod
    def _spacy_compute_a3a4(cls, token: Token) -> str:
        """Function to compute A3 and A4

        Parameters
        ----------
        token : Token

        Returns
        -------
        str
        """

        if token.is_upper:
            return "UPPER"

        elif token.shape_.startswith("Xx"):
            return "S_UPPER"

        elif token.shape_.startswith("x"):
            return "LOWER"

        elif (token.is_digit) & (
            (token.doc[max(token.i - 1, 0)].is_punct)
            | (token.doc[min(token.i + 1, len(token.doc) - 1)].is_punct)
        ):
            return "ENUMERATION"

        elif token.is_digit:
            return "DIGIT"

        elif (token.is_punct) & (token.text in [".", ";", "..", "..."]):
            return "STRONG_PUNCT"

        elif (token.is_punct) & (token.text not in [".", ";", "..", "..."]):
            return "SOFT_PUNCT"

        else:
            return "OTHER"

    @classmethod
    def _compute_length(cls, doc: Doc, start: int, end: int) -> int:
        """Compute length without spaces

        Parameters
        ----------
        doc : Doc
        start : int
        end : int

        Returns
        -------
        int
        """
        length = 0
        for t in doc[start:end]:
            length += len(t.text)

        return length

    def _get_df(self, doc: Doc, new_lines: List[Span]) -> pd.DataFrame:
        """Get a pandas DataFrame to call the classifier

        Parameters
        ----------
        doc : Doc
        new_lines : List[Span]

        Returns
        -------
        pd.DataFrame
        """

        data = []
        for i, span in enumerate(new_lines):
            start = span.start
            end = span.end

            max_index = len(doc) - 1
            a1_token = doc[max(start - 1, 0)]
            a2_token = doc[min(start + 1, max_index)]
            a1 = a1_token.orth
            a2 = a2_token.orth
            a3 = self._spacy_compute_a3a4(a1_token)
            a4 = self._spacy_compute_a3a4(a2_token)
            blank_line = "\n\n" in span.text

            if i > 0:
                start_previous = new_lines[i - 1].start + 1
            else:
                start_previous = 0

            length = self._compute_length(
                doc, start=start_previous, end=start
            )  # It's ok cause i count the total length from the previous up to this one

            data_dict = dict(
                span_start=start,
                span_end=end,
                A1=a1,
                A2=a2,
                A3=a3,
                A4=a4,
                BLANK_LINE=blank_line,
                length=length,
            )
            data.append(data_dict)

        df = pd.DataFrame(data)

        mu = df["length"].mean()
        sigma = df["length"].std()
        if np.isnan(sigma):
            sigma = 1

        cv = sigma / mu
        df["B1"] = (df["length"] - mu) / sigma
        df["B2"] = cv

        return df

    def __call__(self, doc: Doc) -> Doc:
        """
        Predict for each new line if it's an end of line or a space.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, with each new line annotated
        """

        matches = self.process(doc)
        new_lines = get_spans(matches, "new_line")

        if len(new_lines) > 0:
            df = self._get_df(doc=doc, new_lines=new_lines)
            df = self.model.predict(df)

            for span, prediction in zip(new_lines, df.PREDICTED_END_LINE):

                for t in span:
                    t.tag_ = "ENDLINE" if prediction else "EXCLUDED"
                    if prediction:
                        t._.excluded = True

        return doc
