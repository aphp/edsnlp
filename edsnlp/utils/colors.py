from itertools import cycle
from typing import Dict, List

# The category20 colour palette
CATEGORY20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


def create_colors(labels: List[str]) -> Dict[str, str]:
    """
    Assign a colour for each label, using category20 palette.
    The method loops over the colour palette in case there are too many labels.

    Parameters
    ----------
    labels : List[str]
        List of labels to colorise in displacy.

    Returns
    -------
    Dict[str, str]
        A displacy-compatible colour assignment.
    """

    colors = {label: cat for label, cat in zip(labels, cycle(CATEGORY20))}

    return colors
