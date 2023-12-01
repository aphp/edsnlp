import pandas as pd
import re

def export_pandas_to_brat(
    ann_path,
    txt_path,
    df_to_convert,
    label_column_name,
    span_column_name,
    term_column_name,
    annotation_column_name = None,
):
    """
    - ann_path: str path where to write the ann file.
    - txt_path: str path where is stored the txt linked to ann file. Useful to check if there are newlines.
    - df_to_convert: Pandas df containing at least a column of labels, a column of spans and a column of terms.
    - label_column_name: str name of the column in df_to_convert containing the labels. This column should be filled with str only.
    - span_column_name: str name of the column in df_to_convert containing the spans. This column should be filled with lists only,
    first element of each list being the beginning of the span and second element being the end.
    - term_column_name: str name of the column in df_to_convert containing the raw str from the raw text. This column should be filled with str only.
    - annotation_column_name: OPTIONAL str name of the column in df_to_convert containing the annotations. This column should be filled with str only.
    If None, no annotation will be saved.
    """
    
    SEP = "\t"
    ANNOTATION_LABEL = "AnnotatorNotes"
    brat_raw = ""
    n_annotation = 0
    
    with open(txt_path, "r") as f:
        txt_raw = f.read()
    
    if annotation_column_name:
        df_to_convert = df_to_convert[[label_column_name, span_column_name, term_column_name, annotation_column_name]]
    else:
        # Create an empty annotation column so that we can iter
        # In a generic pandas dataframe
        df_to_convert = df_to_convert[[label_column_name, span_column_name, term_column_name]]
        df_to_convert[annotation_column_name] = ""
        
    # Iter through df to write each line of ann file
    for index, (label, span, term, annotation) in df_to_convert.iterrows():
        term_raw = txt_raw[span[0]:span[1]]
        if "\n" in term_raw:
            span_str = str(span[0]) + "".join(" " + str(span[0] + newline_index.start()) + ";" + str(span[0] + newline_index.start() + 1) for newline_index in re.finditer("\n", term_raw)) + " " + str(span[1])
        else:
            span_str = str(span[0]) + " " + str(span[1])
        brat_raw += "T" + str(index+1) + SEP + label + " " + span_str + SEP + term + "\n"
        if len(annotation):
            n_annotation += 1
            brat_raw += "#" + str(n_annotation) + SEP + ANNOTATION_LABEL + " " + "T" + str(index+1) + SEP + annotation + "\n"
    
    brat_raw = brat_raw[:-2]
    with open(ann_path, "w") as f:
        print(brat_raw, file=f)
