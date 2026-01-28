"""
Evaluation module: Deterministic evaluation harness with fixed seeds.

Scientific role:
- Provides controlled evaluation environment
- Implements held-out test set protocol
- Computes robust metrics
- Prevents evaluation-time learning
"""

from .evaluator import Evaluator

__all__ = ["Evaluator"]
