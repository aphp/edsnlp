import copy

from edsnlp.pipes.core.contextual_matcher.models import FullConfig

healthy_status_dict = dict(
    before=[
        "preservation",
    ],
    after=[
        "conservee?",
        r"\bnormale?",
        "preservee?s?",
        r"\bbien\b",
    ],
    both=[
        r"\bbon(?:ne)?\b",
        r"\bcorrecte?",
    ],
)

altered_status_dict = dict(
    before=[
        "degradation",
        "alteration",
    ],
    after=[
        "degradee?",
        "alteree?",
        r"\bmal\b",
        "anormale?",
    ],
    both=[
        "mauvaise?",
        "troubles?",
        "diminution",
        "deficit",
    ],
)

HEALTHY_STATUS_COMPLEMENTS = (
    healthy_status_dict["before"]
    + healthy_status_dict["after"]
    + healthy_status_dict["both"]
)


ALTERED_STATUS_COMPLEMENTS = (
    altered_status_dict["before"]
    + altered_status_dict["after"]
    + altered_status_dict["both"]
)


def make_assign_regex(complement_list):
    """Format a list of regex to make an 'assign' parameter.

    This function merges multiple regex into one big 'OR' matching group,
    that can be used as the 'assign' parameter of a contextual matcher's pattern.
    """
    return rf"({'|'.join(complement for complement in complement_list)})"


def make_status_assign(
    before: int = -3,
    after: int = 3,
    priority: bool = True,
    altered_level: str = "altered",
):
    """Function to create common assign dictionnaries.

    The priority argument serves to indicate whether the assign dict should have
    priority on the initial regex regarding severity status."""

    if priority:
        healthy, altered = "healthy", altered_level
    else:
        healthy, altered = "good", "bad"
    return [
        dict(
            name=f"{healthy}_status_before",
            regex=make_assign_regex(healthy_status_dict["before"]),
            window=before,
        ),
        dict(
            name=f"{healthy}_status_after",
            regex=make_assign_regex(healthy_status_dict["after"]),
            window=after,
        ),
        dict(
            name=f"{healthy}_status_both",
            regex=make_assign_regex(healthy_status_dict["both"]),
            window=(before, after),
        ),
        dict(
            name=f"{altered}_status_before",
            regex=make_assign_regex(altered_status_dict["before"]),
            window=before,
        ),
        dict(
            name=f"{altered}_status_after",
            regex=make_assign_regex(altered_status_dict["after"]),
            window=after,
        ),
        dict(
            name=f"{altered}_status_both",
            regex=make_assign_regex(altered_status_dict["both"]),
            window=(before, after),
        ),
    ]


def make_include_dict_from_list(list_dict):
    """Function to merge several dictionnaries into one suitable for an 'include'.

    The typical use-case is when multiple 'assigns' are possible, and we want
    to return a match only when at least one of these 'assigns' is matched, ie
    we make the 'include' an OR statement.
    If we were to have the 'include' be a list of dictionnaries similar to the
    'assign' list, it would instead be an AND statement."""
    regex_list = []
    window = [0, 0]
    for include in list_dict:
        current_regex = include["regex"].lstrip("(").rstrip(")")
        current_window = include["window"]
        regex_list.append(current_regex)
        if isinstance(current_window, int):
            if current_window < window[0]:
                window[0] = current_window
            if current_window > window[1]:
                window[1] = current_window
        elif isinstance(current_window, tuple):
            if current_window[0] < window[0]:
                window[0] = current_window[0]
            if current_window[1] > window[1]:
                window[1] = current_window[1]
    new_regex = make_assign_regex(regex_list)
    return dict(regex=new_regex, window=tuple(window), regex_attr="NORM")


def normalize_space_characters(patterns: FullConfig):
    """Function to normalize space characters in regex.

    This function can be used to keep regex definitions human-readable,
    while still processing correctly spaces.
    This function is useful since during development, it was found out some
    specific edge cases required the domain matchers to set the 'ignore_space_tokens'
    argument of their ContextualMatcher to False."""
    normalized_patterns = []
    for config in patterns:
        normalized_regex = [
            r.replace(" ", r"\s{1,3}").replace("'", r"'\s?") for r in config.regex
        ]
        normalized_regex_dict = copy.deepcopy(config)
        normalized_regex_dict.regex = normalized_regex
        normalized_patterns.append(normalized_regex_dict)
    return normalized_patterns
