"""Prompt schema and management system with versioning and validation."""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PolicyParameters:
    """Mutable policy parameters for decision making."""
    health_threshold: float = 0.3  # Flee if health below
    hunger_threshold: float = 0.5  # Eat if hunger above
    max_exploration_steps: int = 100
    max_combat_attempts: int = 3
    risk_tolerance: float = 0.5  # 0=conservative, 1=aggressive
    skill_priority: List[str] = field(default_factory=lambda: [
        "explore", "fight", "eat", "use_item", "flee"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyParameters':
        """Create from dict."""
        return cls(**data)


@dataclass
class PromptSchema:
    """
    Structured prompt with frozen and mutable components.
    
    Decomposition:
    - version: Semantic version string (v1.0.0, v1.0.1, etc.)
    - timestamp: Creation/modification timestamp
    - task_definition: Problem statement (frozen)
    - rules_constraints: Behavioral rules (mostly frozen, immutable)
    - skill_descriptions: Available actions (frozen)
    - policy_parameters: Mutable decision logic
    
    This design enables:
    - Reproducible mutations (all changes tracked)
    - Rollback to any previous version
    - Controlled policy optimization
    - No free-text rewriting (only structured mutations)
    """
    
    version: str  # Semantic version (v1.0.0, v1.1.0, etc.)
    timestamp: str  # ISO format creation time
    task_definition: str  # Frozen: "You are a NetHack player"
    rules_constraints: List[str]  # Frozen: behavioral rules
    skill_descriptions: Dict[str, str]  # Frozen: skill definitions
    policy_parameters: PolicyParameters  # Mutable: decision thresholds, priorities
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)  # Audit trail
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'task_definition': self.task_definition,
            'rules_constraints': self.rules_constraints,
            'skill_descriptions': self.skill_descriptions,
            'policy_parameters': self.policy_parameters.to_dict(),
            'mutation_history': self.mutation_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptSchema':
        """Create from dict."""
        return cls(
            version=data['version'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            task_definition=data['task_definition'],
            rules_constraints=data['rules_constraints'],
            skill_descriptions=data['skill_descriptions'],
            policy_parameters=PolicyParameters.from_dict(data['policy_parameters']),
            mutation_history=data.get('mutation_history', []),
        )
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'PromptSchema':
        """Create from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    def to_prompt_string(self) -> str:
        """
        Convert to prompt string for LLM.
        
        Format:
        TASK DEFINITION
        [task_definition]
        
        RULES AND CONSTRAINTS
        - [rule 1]
        - [rule 2]
        ...
        
        AVAILABLE SKILLS
        - skill_name: description
        ...
        
        DECISION POLICY PARAMETERS
        {json policy parameters}
        
        RECENT MUTATIONS
        [mutation history if any]
        """
        prompt = f"""TASK DEFINITION
{self.task_definition}

RULES AND CONSTRAINTS
"""
        for rule in self.rules_constraints:
            prompt += f"- {rule}\n"
        
        prompt += "\nAVAILABLE SKILLS\n"
        for skill_name, description in self.skill_descriptions.items():
            prompt += f"- {skill_name}: {description}\n"
        
        prompt += "\nDECISION POLICY PARAMETERS\n"
        prompt += json.dumps(self.policy_parameters.to_dict(), indent=2)
        
        if self.mutation_history:
            prompt += "\n\nRECENT MUTATIONS\n"
            for mutation in self.mutation_history[-5:]:  # Last 5 mutations
                prompt += f"- {mutation['description']} (v{mutation['version']})\n"
        
        return prompt
    
    def validate(self) -> bool:
        """Validate prompt schema."""
        # Check required fields
        if not self.version or not self.task_definition:
            logger.error("Missing required fields")
            return False
        
        # Check version format
        if not self._is_valid_version(self.version):
            logger.error(f"Invalid version format: {self.version}")
            return False
        
        # Check policy parameters
        params = self.policy_parameters
        if not (0 <= params.health_threshold <= 1):
            logger.error("health_threshold must be in [0, 1]")
            return False
        if not (0 <= params.hunger_threshold <= 1):
            logger.error("hunger_threshold must be in [0, 1]")
            return False
        if not (0 <= params.risk_tolerance <= 1):
            logger.error("risk_tolerance must be in [0, 1]")
            return False
        
        # Check skill descriptions exist
        if not self.skill_descriptions:
            logger.error("No skill descriptions provided")
            return False
        
        return True
    
    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version string is valid (semantic versioning)."""
        parts = version.lstrip('v').split('.')
        if len(parts) != 3:
            return False
        try:
            return all(p.isdigit() for p in parts)
        except:
            return False
    
    def add_mutation(self, mutation_description: str, prev_version: str) -> None:
        """Record a mutation in history."""
        self.mutation_history.append({
            'version': self.version,
            'prev_version': prev_version,
            'description': mutation_description,
            'timestamp': datetime.now().isoformat(),
        })


class PromptManager:
    """
    Manage prompt versioning, storage, and mutations.
    
    Design:
    - All versions stored on disk
    - Versioning follows semantic versioning (v1.0.0, v1.0.1, v1.1.0, v2.0.0)
    - Full mutation history tracked
    - Rollback to any previous version supported
    - No free-text mutations (only structured operators)
    """
    
    def __init__(self, prompt_dir: str):
        """
        Initialize prompt manager.
        
        Args:
            prompt_dir: Directory for prompt storage
        """
        self.prompt_dir = Path(prompt_dir)
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.current_prompt: Optional[PromptSchema] = None
        self.version_index: Dict[str, PromptSchema] = {}  # In-memory index
        self._load_all_versions()
    
    def _load_all_versions(self) -> None:
        """Load all versions from disk into memory."""
        yaml_files = sorted(self.prompt_dir.glob("prompt_v*.yaml"))
        for filepath in yaml_files:
            try:
                with open(filepath, 'r') as f:
                    prompt = PromptSchema.from_yaml(f.read())
                    self.version_index[prompt.version] = prompt
                    logger.debug(f"Loaded prompt {prompt.version}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    
    def save_prompt(self, prompt: PromptSchema) -> Path:
        """
        Save prompt version to disk.
        
        Args:
            prompt: Prompt to save
            
        Returns:
            Path to saved file
        """
        if not prompt.validate():
            raise ValueError(f"Invalid prompt: {prompt.version}")
        
        filepath = self.prompt_dir / f"prompt_{prompt.version}.yaml"
        
        # Prevent overwriting (immutability)
        if filepath.exists():
            raise ValueError(f"Prompt {prompt.version} already exists (immutable)")
        
        with open(filepath, 'w') as f:
            f.write(prompt.to_yaml())
        
        self.version_index[prompt.version] = prompt
        logger.info(f"Saved prompt version {prompt.version} to {filepath}")
        return filepath
    
    def load_prompt(self, version: str) -> Optional[PromptSchema]:
        """
        Load prompt by version string.
        
        Args:
            version: Version string (e.g., "v1.0.0")
            
        Returns:
            PromptSchema or None if not found
        """
        if version in self.version_index:
            return self.version_index[version]
        
        logger.error(f"Prompt version {version} not found")
        return None
    
    def set_current_prompt(self, prompt: PromptSchema) -> None:
        """
        Set current prompt (and save to disk).
        
        Args:
            prompt: Prompt to set as current
        """
        self.save_prompt(prompt)
        self.current_prompt = prompt
        logger.info(f"Set current prompt to version {prompt.version}")
        
        # Save current version index to file
        self._save_version_index()
    
    def _save_version_index(self) -> None:
        """Save version index to file for reference."""
        index_path = self.prompt_dir / "versions.json"
        index_data = {
            'current': self.current_prompt.version if self.current_prompt else None,
            'all_versions': list(self.version_index.keys()),
        }
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def get_current_prompt(self) -> Optional[PromptSchema]:
        """Get current prompt."""
        return self.current_prompt
    
    def list_versions(self) -> List[str]:
        """List all available prompt versions."""
        return sorted(self.version_index.keys())
    
    def rollback(self, target_version: str) -> bool:
        """
        Rollback to previous version.
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            True if successful
        """
        if target_version not in self.version_index:
            logger.error(f"Cannot rollback: version {target_version} not found")
            return False
        
        prompt = self.version_index[target_version]
        self.current_prompt = prompt
        self._save_version_index()
        logger.info(f"Rolled back to version {target_version}")
        return True
    
    def get_mutation_history(self) -> List[Dict[str, Any]]:
        """Get full mutation history of current prompt."""
        if not self.current_prompt:
            return []
        return self.current_prompt.mutation_history


def create_base_prompt() -> PromptSchema:
    """
    Create the base/initial prompt (v1.0.0).
    
    This is the foundation for all experiments. It defines:
    - The core task (NetHack playing)
    - Inviolable rules (frozen)
    - Skill set (frozen)
    - Initial policy parameters (mutable)
    """
    return PromptSchema(
        version="v1.0.0",
        timestamp=datetime.now().isoformat(),
        task_definition="""You are an expert NetHack player. Your goal is to survive as long as possible 
and maximize your score. Make strategic, tactical decisions about movement, combat, inventory management, 
and skill usage based on the current game state and your policy parameters.""",
        rules_constraints=[
            "Never waste critical resources (potions, scrolls) on trivial threats",
            "Always maintain health above the health_threshold before engaging in combat",
            "When hungry, eat food from inventory before exploring dangerous areas",
            "Explore systematically, mapping dungeon layout for strategic advantage",
            "Flee or rest when injured, do not fight at low health",
            "Use available items strategically (not randomly)",
            "Track inventory space and avoid unnecessary items",
        ],
        skill_descriptions={
            "explore": "Navigate to unexplored areas, move through dungeon, discover resources and enemies",
            "fight": "Engage nearby enemies using available weapons and combat tactics",
            "flee": "Escape from danger or retreat to safer areas for recovery",
            "eat": "Consume food from inventory to restore hunger/health status",
            "use_item": "Use special items from inventory (potions, scrolls, wands, etc.) strategically",
        },
        policy_parameters=PolicyParameters(
            health_threshold=0.3,
            hunger_threshold=0.5,
            max_exploration_steps=100,
            max_combat_attempts=3,
            risk_tolerance=0.5,
            skill_priority=["explore", "fight", "eat", "use_item", "flee"],
        ),
        mutation_history=[],
    )
