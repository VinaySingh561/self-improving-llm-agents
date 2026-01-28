"""Prompt mutation operators for structured self-improvement.

Design Principles:
- No free-text rewriting (all mutations are explicit, structured operations)
- All mutations are reversible (can rollback to any version)
- All mutations are recorded in prompt history
- Mutations are version-tracked (each creates new semantic version)
- Only policy parameters are mutable (task/rules/skills are frozen)
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of structured mutations."""
    THRESHOLD_ADJUST = "threshold_adjustment"
    SKILL_PRIORITY = "skill_priority_reorder"
    RISK_TOLERANCE = "risk_tolerance_adjust"
    CONSTRAINT_RELAX = "constraint_relaxation"
    PARAMETER_SWEEP = "parameter_sweep"


class MutationOperator(ABC):
    """
    Abstract base for structured prompt mutations.
    
    Contract:
    - Mutations are explicit and deterministic
    - Mutations are fully logged
    - Mutations create new versioned prompt
    - Mutations can be reversed via version history
    """
    
    def __init__(self, name: str, mutation_type: MutationType):
        self.name = name
        self.mutation_type = mutation_type
    
    @abstractmethod
    def apply(
        self,
        current_params: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Apply mutation to policy parameters.
        
        Args:
            current_params: Current policy parameters
            **kwargs: Mutation-specific arguments
            
        Returns:
            (mutated_params, mutation_description)
        """
        pass
    
    def __call__(self, current_params: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], str]:
        """Allow operator to be called as function."""
        return self.apply(current_params, **kwargs)


class ThresholdAdjustmentMutation(MutationOperator):
    """
    Adjust decision thresholds (health, hunger, etc.).
    
    Example:
    - Increase health_threshold to be more conservative (flee earlier)
    - Decrease hunger_threshold to eat less frequently
    """
    
    def __init__(self, threshold_name: str):
        super().__init__(
            name=f"threshold_{threshold_name}",
            mutation_type=MutationType.THRESHOLD_ADJUST
        )
        self.threshold_name = threshold_name
    
    def apply(
        self,
        current_params: Dict[str, Any],
        delta: float = 0.1,
        direction: str = "increase"
    ) -> Tuple[Dict[str, Any], str]:
        """
        Adjust threshold by delta.
        
        Args:
            current_params: Current parameters
            delta: Magnitude of adjustment (0.0-1.0)
            direction: "increase" or "decrease"
            
        Returns:
            (mutated_params, description)
        """
        if self.threshold_name not in current_params:
            logger.warning(f"Threshold {self.threshold_name} not found")
            return current_params, f"Threshold {self.threshold_name} not found"
        
        new_params = current_params.copy()
        old_value = new_params[self.threshold_name]
        
        # Apply direction
        adjustment = delta if direction == "increase" else -delta
        new_value = max(0.0, min(1.0, old_value + adjustment))
        
        new_params[self.threshold_name] = new_value
        
        description = (
            f"Adjusted {self.threshold_name}: {old_value:.3f} → {new_value:.3f} "
            f"({direction}, Δ={delta:.3f})"
        )
        
        logger.info(f"Mutation: {description}")
        return new_params, description


class RiskToleranceMutation(MutationOperator):
    """Adjust risk tolerance (0=conservative, 1=aggressive)."""
    
    def __init__(self):
        super().__init__(
            name="risk_tolerance",
            mutation_type=MutationType.RISK_TOLERANCE
        )
    
    def apply(
        self,
        current_params: Dict[str, Any],
        new_risk: float = 0.5
    ) -> Tuple[Dict[str, Any], str]:
        """
        Set risk tolerance to specific value.
        
        Args:
            current_params: Current parameters
            new_risk: New risk tolerance (0.0-1.0)
            
        Returns:
            (mutated_params, description)
        """
        new_params = current_params.copy()
        old_risk = new_params.get('risk_tolerance', 0.5)
        new_risk = max(0.0, min(1.0, new_risk))
        new_params['risk_tolerance'] = new_risk
        
        risk_labels = {
            0.0: "very conservative",
            0.25: "conservative",
            0.5: "balanced",
            0.75: "aggressive",
            1.0: "very aggressive",
        }
        
        old_label = risk_labels.get(round(old_risk * 4) / 4, f"{old_risk:.2f}")
        new_label = risk_labels.get(round(new_risk * 4) / 4, f"{new_risk:.2f}")
        
        description = f"Changed risk tolerance: {old_label} ({old_risk:.3f}) → {new_label} ({new_risk:.3f})"
        logger.info(f"Mutation: {description}")
        return new_params, description


class SkillPriorityMutation(MutationOperator):
    """
    Reorder skill priority list.
    
    Example:
    - Move "fight" higher if agent is dying to enemies
    - Move "eat" higher if agent is starving
    """
    
    def __init__(self):
        super().__init__(
            name="skill_priority",
            mutation_type=MutationType.SKILL_PRIORITY
        )
    
    def apply(
        self,
        current_params: Dict[str, Any],
        operation: str = "rotate",
        skill: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Modify skill priority order.
        
        Args:
            current_params: Current parameters
            operation: "rotate" | "prioritize" | "deprioritize"
            skill: Specific skill to prioritize/deprioritize
            
        Returns:
            (mutated_params, description)
        """
        new_params = current_params.copy()
        old_priority = new_params.get('skill_priority', []).copy()
        new_priority = old_priority.copy()
        
        if operation == "rotate":
            # Move last skill to front
            if len(new_priority) > 1:
                new_priority = [new_priority[-1]] + new_priority[:-1]
            description = f"Rotated skill priority: {old_priority} → {new_priority}"
        
        elif operation == "prioritize" and skill:
            # Move skill to front
            if skill in new_priority:
                new_priority.remove(skill)
                new_priority.insert(0, skill)
                description = f"Prioritized '{skill}': {old_priority} → {new_priority}"
            else:
                return new_params, f"Skill '{skill}' not in priority list"
        
        elif operation == "deprioritize" and skill:
            # Move skill to back
            if skill in new_priority:
                new_priority.remove(skill)
                new_priority.append(skill)
                description = f"Deprioritized '{skill}': {old_priority} → {new_priority}"
            else:
                return new_params, f"Skill '{skill}' not in priority list"
        
        else:
            return new_params, "Invalid operation"
        
        new_params['skill_priority'] = new_priority
        logger.info(f"Mutation: {description}")
        return new_params, description


class MaxStepsMutation(MutationOperator):
    """Adjust maximum steps for skill execution."""
    
    def __init__(self):
        super().__init__(
            name="max_steps",
            mutation_type=MutationType.PARAMETER_SWEEP
        )
    
    def apply(
        self,
        current_params: Dict[str, Any],
        exploration_steps: Optional[int] = None,
        combat_attempts: Optional[int] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Adjust max steps/attempts for skills.
        
        Args:
            current_params: Current parameters
            exploration_steps: New max exploration steps
            combat_attempts: New max combat attempts
            
        Returns:
            (mutated_params, description)
        """
        new_params = current_params.copy()
        changes = []
        
        if exploration_steps is not None:
            old = new_params.get('max_exploration_steps', 100)
            new_params['max_exploration_steps'] = max(10, exploration_steps)
            changes.append(f"exploration_steps: {old} → {new_params['max_exploration_steps']}")
        
        if combat_attempts is not None:
            old = new_params.get('max_combat_attempts', 3)
            new_params['max_combat_attempts'] = max(1, combat_attempts)
            changes.append(f"combat_attempts: {old} → {new_params['max_combat_attempts']}")
        
        description = f"Adjusted limits: {', '.join(changes)}"
        logger.info(f"Mutation: {description}")
        return new_params, description


class MutationProposer:
    """
    Generate candidate mutations from failure analysis.
    
    Design:
    - Failures trigger specific mutation suggestions
    - All mutations are deterministic and logged
    - Mutations are ranked by relevance to failures
    """
    
    def __init__(self):
        self.mutation_operators = {
            "health_threshold": ThresholdAdjustmentMutation("health_threshold"),
            "hunger_threshold": ThresholdAdjustmentMutation("hunger_threshold"),
            "risk_tolerance": RiskToleranceMutation(),
            "skill_priority": SkillPriorityMutation(),
            "max_steps": MaxStepsMutation(),
        }
        self.mutation_history = []
    
    def propose_mutations(
        self,
        failure_analysis: Dict[str, Any],
        current_params: Dict[str, Any],
        num_mutations: int = 5
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        Propose mutations based on failure patterns.
        
        Args:
            failure_analysis: Failure classifications from meta-learner
            current_params: Current policy parameters
            num_mutations: Number of mutations to generate
            
        Returns:
            List of (mutated_params, description) tuples
        """
        mutations = []
        
        # Analyze failures and propose targeted mutations
        failure_counts = failure_analysis.get('failure_counts', {})
        
        # If many injury failures: increase health_threshold (more conservative)
        if failure_counts.get('injury', 0) > 2:
            op = ThresholdAdjustmentMutation('health_threshold')
            params, desc = op.apply(current_params, delta=0.1, direction="increase")
            mutations.append((params, desc))
            logger.info(f"Proposed mutation (injury prevention): {desc}")
        
        # If many starvation failures: decrease hunger_threshold (eat earlier)
        if failure_counts.get('starvation', 0) > 2:
            op = ThresholdAdjustmentMutation('hunger_threshold')
            params, desc = op.apply(current_params, delta=0.1, direction="decrease")
            mutations.append((params, desc))
            logger.info(f"Proposed mutation (hunger prevention): {desc}")
        
        # If many combat losses: prioritize escape/food
        if failure_counts.get('combat_loss', 0) > 1:
            op = SkillPriorityMutation()
            params, desc = op.apply(current_params, operation="prioritize", skill="flee")
            mutations.append((params, desc))
            logger.info(f"Proposed mutation (combat avoidance): {desc}")
        
        # If many entrapment failures: increase exploration steps
        if failure_counts.get('entrapment', 0) > 1:
            op = MaxStepsMutation()
            params, desc = op.apply(current_params, exploration_steps=150)
            mutations.append((params, desc))
            logger.info(f"Proposed mutation (better exploration): {desc}")
        
        # Add baseline mutations (always explore alternatives)
        # Conservative version
        if len(mutations) < num_mutations:
            op = RiskToleranceMutation()
            params, desc = op.apply(current_params, new_risk=max(0, current_params.get('risk_tolerance', 0.5) - 0.25))
            mutations.append((params, desc))
        
        # Aggressive version
        if len(mutations) < num_mutations:
            op = RiskToleranceMutation()
            params, desc = op.apply(current_params, new_risk=min(1, current_params.get('risk_tolerance', 0.5) + 0.25))
            mutations.append((params, desc))
        
        # Skill priority rotation
        if len(mutations) < num_mutations:
            op = SkillPriorityMutation()
            params, desc = op.apply(current_params, operation="rotate")
            mutations.append((params, desc))
        
        # Return top N mutations
        return mutations[:num_mutations]
    
    def get_mutation_history(self) -> List[Dict[str, Any]]:
        """Get history of all proposed mutations."""
        return self.mutation_history

