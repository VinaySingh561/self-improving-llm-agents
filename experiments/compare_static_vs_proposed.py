import numpy as np
from scipy.stats import mannwhitneyu

from experiments.aggregate_static import all_scores as static_scores
from experiments.aggregate_proposed import all_scores as proposed_scores


static = np.array(static_scores)
proposed = np.array(proposed_scores)

print("STATISTICAL COMPARISON: STATIC vs PROPOSED")
print("=" * 50)

# Mann–Whitney U test
u_stat, p_value = mannwhitneyu(
    proposed,
    static,
    alternative="two-sided"
)

print(f"Mann–Whitney U statistic: {u_stat:.2f}")
print(f"p-value: {p_value:.4f}")

# Effect size (Cohen's d)
mean_diff = proposed.mean() - static.mean()
pooled_std = np.sqrt(
    (static.std() ** 2 + proposed.std() ** 2) / 2
)
cohens_d = mean_diff / pooled_std

print(f"Cohen's d: {cohens_d:.3f}")

# Interpretation
print("\nINTERPRETATION")
if p_value < 0.05:
    print("✓ Statistically significant difference (p < 0.05)")
else:
    print("✗ No statistically significant difference (p ≥ 0.05)")

if abs(cohens_d) < 0.2:
    print("Effect size: negligible")
elif abs(cohens_d) < 0.5:
    print("Effect size: small")
else:
    print("Effect size: medium or larger")
