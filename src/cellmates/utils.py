from tdm.cell_types import CELL_TYPES_ARRAY


def microns(x: int) -> float:
    """Converts the number of microns x, into a float representing the number of meters.

    Example:
        microns(1) = 1e-6

    Args:
        x (int): _description_

    Returns:
        float: _description_
    """
    return x * 1e-6


# maximal effective communication distance based on euler-yaniv 2023
MAX_EFFECTIVE_DISTANCE = 140

CELL_TYPE_STR_TO_IDX = {k: i+1 for i, k in enumerate(CELL_TYPES_ARRAY)}
CELL_TYPE_IDX_TO_STR = {i+1: k for i, k in enumerate(CELL_TYPES_ARRAY)}


N_CELL_TYPES = 7 # special cell type + six main types
