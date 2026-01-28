"""
Prompts module: Machine-editable prompt system with versioning and mutation.

Scientific role:
- Decomposes prompts into frozen and mutable components
- Enables controlled, reproducible prompt evolution
- Tracks all prompt changes with explicit versions
- Prevents free-text rewriting (mutations are structured)

Design:
- PromptSchema: Core data structure with decomposed components
- PromptManager: Versioning, storage, and rollback
- MutationOperator: Structured mutation implementations
- MutationProposer: Intelligent mutation generation from failure analysis
"""

from .prompt_schema import (
    PromptSchema,
    PolicyParameters,
    PromptManager,
    create_base_prompt,
)
from .mutation_operators import (
    MutationOperator,
    MutationType,
    ThresholdAdjustmentMutation,
    RiskToleranceMutation,
    SkillPriorityMutation,
    MaxStepsMutation,
    MutationProposer,
)

__all__ = [
    "PromptSchema",
    "PolicyParameters",
    "PromptManager",
    "create_base_prompt",
    "MutationOperator",
    "MutationType",
    "ThresholdAdjustmentMutation",
    "RiskToleranceMutation",
    "SkillPriorityMutation",
    "MaxStepsMutation",
    "MutationProposer",
]
