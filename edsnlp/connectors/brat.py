import warnings
from pathlib import Path
from typing import List, Optional, Union

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol
from edsnlp.data.converters import AttributesMappingArg
from edsnlp.data.standoff import (
    dump_standoff_file,
    parse_standoff_file,
    read_standoff,
    write_standoff,
)
from edsnlp.utils.span_getters import (
    SpanSetterArg,
    validate_span_setter,
)


class BratConnector(object):
    """
    Deprecated. Use `edsnlp.data.read_standoff` and `edsnlp.data.write_standoff`
    instead.
    Two-way connector with BRAT. Supports entities only.

    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing the BRAT files.
    n_jobs : int, optional
        Number of jobs for multiprocessing, by default 1
    attributes: Optional[Union[Sequence[str], Mapping[str, str]]]
        Mapping from BRAT attributes to spaCy Span extensions.
        Extensions / attributes that are not in the mapping are not imported or exported
        If left to None, the mapping is filled with all BRAT attributes.
    span_groups: Optional[Sequence[str]]
        Additional span groups to look for entities in spaCy documents when exporting.
        Missing label (resp. span group) names are not imported (resp. exported)
        If left to None, the sequence is filled with all BRAT entity labels.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        n_jobs: int = 1,
        attributes: Optional[AttributesMappingArg] = None,
        bool_attributes: Optional[List[str]] = [],
        span_groups: SpanSetterArg = ["ents", "*"],
        keep_raw_attribute_values: bool = False,
    ):
        warnings.warn(
            "This connector is deprecated and will be removed in a future version.\n"
            "Use `edsnlp.data.read_standoff` and `edsnlp.data.write_standoff` instead.",
            DeprecationWarning,
        )
        self.directory: Path = Path(directory)
        self.attr_map = attributes
        self.span_setter = validate_span_setter(span_groups)
        self.keep_raw_attribute_values = keep_raw_attribute_values
        self.bool_attributes = list(bool_attributes)

    def brat2docs(self, nlp: PipelineProtocol, run_pipe=False) -> List[Doc]:
        res = read_standoff(
            path=self.directory,
            nlp=nlp,
            keep_txt_only_docs=True,
            span_attributes=self.attr_map,
            span_setter=self.span_setter,
            keep_raw_attribute_values=self.keep_raw_attribute_values,
            bool_attributes=self.bool_attributes,
        )
        return list(nlp.pipe(res) if run_pipe else res)

    def docs2brat(self, docs: List[Doc]) -> None:
        """
        Writes a list of spaCy documents to file.
        """
        write_standoff(
            docs,
            self.directory,
            span_getter=self.span_setter,
            span_attributes=self.attr_map or {},
            overwrite=True,
        )


load_from_brat = parse_standoff_file
export_to_brat = dump_standoff_file
