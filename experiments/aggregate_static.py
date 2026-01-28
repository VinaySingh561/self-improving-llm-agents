# import sys
# import os

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from experiments.run_static_baseline import run_static

SEEDS = [42, 123, 456, 789, 999]

all_scores = []
all_alive = []

for seed in SEEDS:
    results = run_static(seed=seed, episodes=10)
    for r in results:
        all_scores.append(r["score"])
        all_alive.append(r["alive"])

scores = np.array(all_scores)
alive = np.array(all_alive)

print("STATIC BASELINE (REAL NLE)")
print("-" * 40)
print(f"Episodes: {len(scores)}")
print(f"Mean score: {scores.mean():.2f}")
print(f"Std score: {scores.std():.2f}")
print(f"Survival rate: {alive.mean():.2%}")
print(f"Min / Max score: {scores.min()} / {scores.max()}")
