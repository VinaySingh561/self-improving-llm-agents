import numpy as np
from typing import List, Tuple


def bootstrap_ci(
    values: List[float],
    num_samples: int = 10_000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval.
    """
    values = np.array(values)
    means = []

    for _ in range(num_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)

    return float(lower), float(upper)
