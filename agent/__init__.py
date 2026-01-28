"""
Agent module: Decision-making entity that orchestrates LLM calls and skill execution.

Scientific role:
- Orchestrates LLM-based decision making at skill boundaries
- Manages internal state between decisions
- Caches LLM decisions to avoid redundant calls
- Provides clean interface to skills

Key Components:
- LLMWrapper: CPU-only quantized LLM inference
- NetHackAgent: Decision orchestration
- AgentStateManager: State serialization
"""

from .nethack_agent import (
    NetHackAgent,
    AgentStateManager,
    AgentState,
    DecisionRecord,
    EpisodeLog,
)
from .llm_wrapper import LLMWrapper, LLMConfig, OutputValidator, LLMCallLog

__all__ = [
    "NetHackAgent",
    "AgentStateManager",
    "AgentState",
    "DecisionRecord",
    "EpisodeLog",
    "LLMWrapper",
    "LLMConfig",
    "OutputValidator",
    "LLMCallLog",
]

