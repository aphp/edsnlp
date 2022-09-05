from typing import List


def flatten(my_list: List):
    """
    Flatten (if necessary) a list of sublists

    Parameters
    ----------
    my_list : List
        A list of items, each items in turn can be a list

    Returns
    -------
    List
        A flatten list
    """
    if not my_list:
        return my_list
    my_list = [item if isinstance(item, list) else [item] for item in my_list]

    return [item for sublist in my_list for item in sublist]
