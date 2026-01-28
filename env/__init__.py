"""
Environment module: NetHack Learning Environment (NLE) initialization and management.

Scientific role:
- Provides deterministic, seed-controlled NetHack instances
- Handles observation/action interface
- Manages episode lifecycle
- Logs environment state for reproducibility
"""

from .nle_wrapper import NLEWrapper

__all__ = ["NLEWrapper"]
