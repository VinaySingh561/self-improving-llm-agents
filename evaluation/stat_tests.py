from typing import List
from scipy.stats import mannwhitneyu
import numpy as np


def compare_methods(
    a: List[float],
    b: List[float],
):
    """
    Mann–Whitney U test at seed level.
    """
    u, p = mannwhitneyu(a, b, alternative="two-sided")

    # Effect size: Cliff's delta (robust, non-parametric)
    delta = cliffs_delta(a, b)

    return {
        "u_stat": float(u),
        "p_value": float(p),
        "cliffs_delta": float(delta),
    }


def cliffs_delta(a, b):
    """
    Effect size for non-parametric comparisons.
    """
    a = np.array(a)
    b = np.array(b)

    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)

    return (gt - lt) / (len(a) * len(b))
