#!/usr/bin/env python3
"""Test and demonstrate the prompt system with versioning and mutations."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.prompt_schema import (
    PromptSchema,
    PolicyParameters,
    PromptManager,
    create_base_prompt,
)
from prompts.mutation_operators import (
    MutationProposer,
    ThresholdAdjustmentMutation,
    SkillPriorityMutation,
    RiskToleranceMutation,
)


def test_base_prompt():
    """Test base prompt creation and validation."""
    print("\n" + "="*70)
    print("TEST 1: Base Prompt Creation and Validation")
    print("="*70)
    
    prompt = create_base_prompt()
    
    print(f"\nVersion: {prompt.version}")
    print(f"Valid: {prompt.validate()}")
    print(f"\nTask Definition:")
    print(f"  {prompt.task_definition[:100]}...")
    print(f"\nRules ({len(prompt.rules_constraints)}):")
    for i, rule in enumerate(prompt.rules_constraints, 1):
        print(f"  {i}. {rule}")
    print(f"\nAvailable Skills ({len(prompt.skill_descriptions)}):")
    for skill, desc in prompt.skill_descriptions.items():
        print(f"  - {skill}: {desc}")
    print(f"\nPolicy Parameters:")
    for key, value in prompt.policy_parameters.to_dict().items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    return prompt


def test_prompt_persistence(base_prompt):
    """Test saving and loading prompts."""
    print("\n" + "="*70)
    print("TEST 2: Prompt Persistence and Versioning")
    print("="*70)
    
    prompt_dir = Path("/tmp/selfimpagents_test/prompts")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manager and save base prompt
    manager = PromptManager(str(prompt_dir))
    manager.set_current_prompt(base_prompt)
    
    print(f"\nSaved prompt to: {prompt_dir}")
    print(f"Available versions: {manager.list_versions()}")
    print(f"Current prompt version: {manager.get_current_prompt().version}")
    
    # Load it back
    loaded = manager.load_prompt("v1.0.0")
    print(f"Loaded prompt: {loaded.version}")
    print(f"Loaded prompt valid: {loaded.validate()}")
    
    return manager


def test_mutations(manager):
    """Test mutation operators."""
    print("\n" + "="*70)
    print("TEST 3: Structured Mutations")
    print("="*70)
    
    current_params = manager.get_current_prompt().policy_parameters.to_dict()
    print(f"\nCurrent parameters:")
    for key, value in current_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- Mutation 1: Increase Health Threshold ---")
    op = ThresholdAdjustmentMutation("health_threshold")
    mutated_params, desc = op.apply(current_params, delta=0.1, direction="increase")
    print(f"Description: {desc}")
    print(f"New health_threshold: {mutated_params['health_threshold']:.3f}")
    
    print(f"\n--- Mutation 2: Prioritize Flee ---")
    op = SkillPriorityMutation()
    mutated_params, desc = op.apply(current_params, operation="prioritize", skill="flee")
    print(f"Description: {desc}")
    print(f"New priority: {mutated_params['skill_priority']}")
    
    print(f"\n--- Mutation 3: Increase Risk Tolerance ---")
    op = RiskToleranceMutation()
    mutated_params, desc = op.apply(current_params, new_risk=0.7)
    print(f"Description: {desc}")
    print(f"New risk_tolerance: {mutated_params['risk_tolerance']:.3f}")


def test_mutation_proposal():
    """Test intelligent mutation proposal from failures."""
    print("\n" + "="*70)
    print("TEST 4: Failure-Driven Mutation Proposal")
    print("="*70)
    
    base_prompt = create_base_prompt()
    current_params = base_prompt.policy_parameters.to_dict()
    
    # Simulate failure analysis
    failure_analysis = {
        'failure_counts': {
            'injury': 5,  # Many injuries
            'starvation': 2,
            'combat_loss': 3,
            'entrapment': 1,
        },
        'total_failures': 11,
    }
    
    print(f"\nFailure Analysis:")
    for failure_type, count in failure_analysis['failure_counts'].items():
        print(f"  {failure_type}: {count}")
    
    proposer = MutationProposer()
    mutations = proposer.propose_mutations(
        failure_analysis,
        current_params,
        num_mutations=5
    )
    
    print(f"\nProposed Mutations ({len(mutations)}):")
    for i, (params, desc) in enumerate(mutations, 1):
        print(f"\n  {i}. {desc}")


def test_version_creation():
    """Test creating new versioned prompts from mutations."""
    print("\n" + "="*70)
    print("TEST 5: Versioning with Mutations")
    print("="*70)
    
    base_prompt = create_base_prompt()
    prompt_dir = Path("/tmp/selfimpagents_test/prompts_v2")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    manager = PromptManager(str(prompt_dir))
    manager.set_current_prompt(base_prompt)
    
    # Create v1.0.1 (minor change)
    v1_01 = PromptSchema(
        version="v1.0.1",
        timestamp=base_prompt.timestamp,
        task_definition=base_prompt.task_definition,
        rules_constraints=base_prompt.rules_constraints,
        skill_descriptions=base_prompt.skill_descriptions,
        policy_parameters=PolicyParameters(
            health_threshold=0.4,  # Changed from 0.3
            hunger_threshold=base_prompt.policy_parameters.hunger_threshold,
            max_exploration_steps=base_prompt.policy_parameters.max_exploration_steps,
            max_combat_attempts=base_prompt.policy_parameters.max_combat_attempts,
            risk_tolerance=base_prompt.policy_parameters.risk_tolerance,
            skill_priority=base_prompt.policy_parameters.skill_priority,
        ),
        mutation_history=[],
    )
    v1_01.add_mutation("Increased health_threshold to be more conservative", "v1.0.0")
    
    manager.set_current_prompt(v1_01)
    
    # Create v1.1.0 (feature/skill change)
    priority = base_prompt.policy_parameters.skill_priority.copy()
    priority.remove("flee")
    priority.insert(0, "flee")  # Prioritize fleeing
    
    v1_10 = PromptSchema(
        version="v1.1.0",
        timestamp=base_prompt.timestamp,
        task_definition=base_prompt.task_definition,
        rules_constraints=base_prompt.rules_constraints,
        skill_descriptions=base_prompt.skill_descriptions,
        policy_parameters=PolicyParameters(
            health_threshold=0.4,
            hunger_threshold=base_prompt.policy_parameters.hunger_threshold,
            max_exploration_steps=120,
            max_combat_attempts=2,
            risk_tolerance=0.3,
            skill_priority=priority,
        ),
        mutation_history=[],
    )
    v1_10.add_mutation("Prioritized flee to reduce combat deaths", "v1.0.1")
    
    manager.set_current_prompt(v1_10)
    
    print(f"\nVersions in order:")
    for version in manager.list_versions():
        print(f"  - {version}")
    
    print(f"\nMutation history of current (v1.1.0):")
    for mutation in manager.get_mutation_history():
        print(f"  - {mutation['description']}")
        print(f"    {mutation['prev_version']} → {mutation['version']}")


def test_prompt_string_rendering():
    """Test rendering prompt to LLM-readable format."""
    print("\n" + "="*70)
    print("TEST 6: Prompt Rendering for LLM")
    print("="*70)
    
    prompt = create_base_prompt()
    prompt_string = prompt.to_prompt_string()
    
    print(f"\nRendered Prompt (first 800 chars):")
    print("-" * 70)
    print(prompt_string[:800])
    print("-" * 70)
    print(f"[... truncated, total length: {len(prompt_string)} chars]")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PROMPT SYSTEM TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Base prompt
        base_prompt = test_base_prompt()
        
        # Test 2: Persistence and versioning
        manager = test_prompt_persistence(base_prompt)
        
        # Test 3: Mutations
        test_mutations(manager)
        
        # Test 4: Proposal
        test_mutation_proposal()
        
        # Test 5: Versioning
        test_version_creation()
        
        # Test 6: Rendering
        test_prompt_string_rendering()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
