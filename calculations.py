import math
import numpy as np


def mean_square_error(a, b):
    """
    Mean Square Error Function
    Arguments:
    a, b - Pandas Series object
    """
    diffs = np.subtract(b, a)
    return 1 / len(a) * sum(diff**2 for diff in diffs)
