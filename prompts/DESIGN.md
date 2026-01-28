# Phase 2: Prompt System Design

## Overview

The prompt system is the **core of self-improvement** in this system. Unlike fine-tuning or gradient updates, all learning occurs through **structured mutations** of a machine-editable prompt.

This design ensures:
- **Reproducibility**: Every mutation is explicit and version-tracked
- **Reversibility**: Rollback to any previous version
- **Interpretability**: All changes are human-readable structured operations
- **Controlled Evolution**: Mutations are driven by external failure analysis, not self-evaluation

---

## Architecture

### 1. Prompt Schema (`PromptSchema`)

The prompt is decomposed into **frozen and mutable components**:

```yaml
version: v1.0.0                    # Semantic version (immutable after creation)
timestamp: 2024-01-21T...          # Creation time (immutable)
task_definition: |                 # FROZEN
  You are an expert NetHack player...

rules_constraints:                 # FROZEN (behavioral contract)
  - Never waste critical resources
  - Always maintain health above threshold
  - Explore systematically
  - ...

skill_descriptions:                # FROZEN (action vocabulary)
  explore: Navigate unexplored areas
  fight: Engage enemies
  flee: Escape danger
  eat: Consume food
  use_item: Use special items

policy_parameters:                 # MUTABLE (only this evolves)
  health_threshold: 0.3            # Flee if health < 30%
  hunger_threshold: 0.5            # Eat if hunger > 50%
  max_exploration_steps: 100
  max_combat_attempts: 3
  risk_tolerance: 0.5              # 0=conservative, 1=aggressive
  skill_priority: [explore, fight, eat, use_item, flee]

mutation_history: []               # AUDIT TRAIL (immutable)
  - version: v1.0.1
    prev_version: v1.0.0
    description: Increased health_threshold
    timestamp: 2024-01-21T...
```

**Key Design Decisions:**

1. **Task and rules are frozen** - Prevents the agent from changing core objectives
2. **Skills are frozen** - The agent cannot invent new actions
3. **Only policy parameters are mutable** - Thresholds, priorities, and tolerances evolve
4. **Mutation history is immutable** - Full audit trail of all changes

---

### 2. Policy Parameters (`PolicyParameters`)

Mutable parameters that control agent behavior:

| Parameter | Range | Meaning |
|-----------|-------|---------|
| `health_threshold` | [0.0, 1.0] | Flee if health drops below this |
| `hunger_threshold` | [0.0, 1.0] | Eat if hunger exceeds this |
| `max_exploration_steps` | int | Max steps for explore skill |
| `max_combat_attempts` | int | Max combat attempts before fleeing |
| `risk_tolerance` | [0.0, 1.0] | 0=conservative, 1=aggressive |
| `skill_priority` | List[str] | Skill execution order |

**Design principle**: These are **interpretable hyperparameters** that can be manually set or evolved.

---

### 3. Prompt Manager (`PromptManager`)

Manages versioning, storage, and retrieval:

```python
manager = PromptManager("/path/to/prompts")

# Save a prompt
manager.set_current_prompt(prompt_v1)

# Load a prompt
prompt = manager.load_prompt("v1.0.0")

# List all versions
versions = manager.list_versions()

# Rollback
manager.rollback("v1.0.0")
```

**Storage strategy**:
- Each version stored as `prompt_v{version}.yaml` (immutable files)
- In-memory index for fast lookup
- Version index file (`versions.json`) for quick reference

**Versioning scheme** (semantic versioning):
- `v1.0.0` - Initial base prompt
- `v1.0.1` - Patch (bug fix, minor threshold adjustment)
- `v1.1.0` - Minor (new feature, reordered skill priorities)
- `v2.0.0` - Major (breaking change to rules/tasks)

---

### 4. Mutation Operators

**Design principle**: No free-text rewriting. All mutations are structured, deterministic operations.

#### `ThresholdAdjustmentMutation`

Adjust decision thresholds:

```python
op = ThresholdAdjustmentMutation("health_threshold")
new_params, desc = op.apply(
    current_params,
    delta=0.1,
    direction="increase"  # or "decrease"
)
# Output: "Adjusted health_threshold: 0.300 → 0.400 (increase, Δ=0.100)"
```

**Use case**: If agent dies frequently → increase health_threshold (more conservative)

#### `SkillPriorityMutation`

Reorder skill execution priority:

```python
op = SkillPriorityMutation()

# Prioritize a specific skill
new_params, desc = op.apply(
    current_params,
    operation="prioritize",
    skill="flee"
)
# Output: "Prioritized 'flee': [...] → ['flee', ...]"

# Deprioritize
new_params, desc = op.apply(
    current_params,
    operation="deprioritize",
    skill="fight"
)

# Rotate priorities
new_params, desc = op.apply(
    current_params,
    operation="rotate"
)
```

**Use case**: If agent keeps dying to combat → prioritize "flee"

#### `RiskToleranceMutation`

Adjust risk attitude:

```python
op = RiskToleranceMutation()
new_params, desc = op.apply(current_params, new_risk=0.7)
# Output: "Changed risk tolerance: balanced (0.500) → aggressive (0.700)"
```

**Use case**: If agent is too passive → increase risk tolerance

#### `MaxStepsMutation`

Adjust skill execution limits:

```python
op = MaxStepsMutation()
new_params, desc = op.apply(
    current_params,
    exploration_steps=150,
    combat_attempts=5
)
```

**Use case**: If agent gets stuck → increase exploration steps

---

### 5. Mutation Proposer (`MutationProposer`)

Intelligently generates mutations from **failure analysis**:

```python
proposer = MutationProposer()

failure_analysis = {
    'failure_counts': {
        'injury': 5,        # Agent dying
        'starvation': 2,    # Hunger depletion
        'combat_loss': 3,   # Losing fights
        'entrapment': 1,    # Getting stuck
    }
}

mutations = proposer.propose_mutations(
    failure_analysis,
    current_params,
    num_mutations=5
)

for params, description in mutations:
    print(f"Proposed: {description}")
```

**Failure-to-mutation mapping**:

| Failure Type | Proposed Mutations |
|--------------|-------------------|
| Many injuries | Increase health_threshold (more conservative) |
| Starvation | Decrease hunger_threshold (eat earlier) |
| Combat losses | Prioritize flee skill |
| Entrapment | Increase exploration steps |

---

## Workflow: From Failure to Mutation to Evaluation

### Step 1: Collect Training Trajectories

```python
# Run N episodes with current prompt
trajectories = []
for seed in train_seeds:
    traj = agent.run_episode(env, seed)
    trajectories.append(traj)
```

### Step 2: Detect and Classify Failures

```python
# Analyze trajectories for failure patterns
detector = FailureDetector()
failures = detector.classify_batch(trajectories)
failure_analysis = detector.analyze_failures()
```

### Step 3: Propose Mutations

```python
# Propose candidates based on failures
proposer = MutationProposer()
mutations = proposer.propose_mutations(
    failure_analysis,
    current_params,
    num_mutations=5
)
```

### Step 4: Create Versioned Prompts

```python
# Each mutation becomes a new version
new_prompts = []
for mutated_params, description in mutations:
    new_prompt = PromptSchema(
        version=f"v1.1.{i}",  # New version
        task_definition=current.task_definition,  # Unchanged
        rules_constraints=current.rules_constraints,  # Unchanged
        skill_descriptions=current.skill_descriptions,  # Unchanged
        policy_parameters=mutated_params,  # Changed
    )
    new_prompt.add_mutation(description, current.version)
    new_prompts.append(new_prompt)
```

### Step 5: Evaluate on Validation Set

```python
# Evaluate each mutation on FROZEN validation seeds
evaluator = Evaluator(train_seeds, val_seeds, test_seeds)
for prompt in new_prompts:
    results = evaluator.evaluate(agent, env, prompt, seed_set="val")
    metrics = compute_metrics(results)
```

### Step 6: Accept Best or Keep Baseline

```python
# Statistical test: Is new mutation better than current?
best_score = metrics_current['mean_score']
best_prompt = current

for prompt, metrics in zip(new_prompts, all_metrics):
    if metrics['mean_score'] > best_score:
        # Apply statistical test
        if mann_whitney_u_test(metrics, current_metrics) < 0.05:
            best_score = metrics['mean_score']
            best_prompt = prompt

# Update current prompt
if best_prompt != current:
    manager.set_current_prompt(best_prompt)
else:
    print("No improvement, keeping current prompt")
```

### Step 7: Repeat or Evaluate on Test Set

- If improving: Repeat mutation cycle
- If stagnant: Stop and evaluate on frozen test set
- Report final results

---

## Example: Complete Mutation Cycle

```yaml
# Iteration 1
Current: v1.0.0
Failure Analysis:
  - injury: 5 (agent dies from low health)
  - combat_loss: 3

Proposed Mutations:
  1. v1.0.1: health_threshold 0.3 → 0.4
  2. v1.0.2: prioritize flee
  3. v1.0.3: risk_tolerance 0.5 → 0.3

Validation Results:
  v1.0.0 (current):  mean_score = 850, std = 120
  v1.0.1:            mean_score = 920, std = 110  ← BEST
  v1.0.2:            mean_score = 880, std = 130
  v1.0.3:            mean_score = 810, std = 140

Statistical Test: v1.0.1 vs v1.0.0
  Mann-Whitney U p-value = 0.032 < 0.05  → Accept v1.0.1

# Iteration 2
Current: v1.0.1
Failure Analysis:
  - starvation: 4 (agent hungry)
  - entrapment: 2

Proposed Mutations:
  1. v1.0.2: hunger_threshold 0.5 → 0.4
  2. v1.0.3: max_exploration_steps 100 → 150
  3. v1.0.4: skill_priority rotate

Validation Results:
  v1.0.1 (current):  mean_score = 920, std = 110
  v1.0.2:            mean_score = 890, std = 125
  v1.0.3:            mean_score = 915, std = 108
  v1.0.4:            mean_score = 910, std = 120

Statistical Test: No mutation significantly better than v1.0.1
  → Keep v1.0.1

# Stop iterating, evaluate v1.0.1 on frozen test set
Test Results: mean_score = 905 ± 95 (95% CI)
```

---

## Why This Design Enables Reproducible Self-Improvement

### 1. No Black-Box Learning

Every change is explicit:
```yaml
v1.0.0 → v1.0.1: Increased health_threshold (0.3 → 0.4) due to injury failures
```

Compare to fine-tuning:
```
LLM weights updated via backprop through 50 million parameters
(Unknown what changed or why)
```

### 2. Full Reversibility

Any version can be restored:
```python
manager.rollback("v1.0.0")
```

This is impossible with gradient-based learning.

### 3. Deterministic Evaluation

Each prompt version is immutable, so evaluation is deterministic.

### 4. Failure-Driven Evolution

Mutations are proposed directly from failure analysis, not from self-evaluation:

```
Failures → Rules → Proposals → Evaluation → Acceptance
    (NOT)
Failures → LLM → Self-Reflection → Changes → Acceptance
```

The second approach (self-evaluation) is susceptible to:
- Hallucination
- Reward hacking
- Compounding errors

---

## Integration with Meta-Learner

The prompt system interfaces with the meta-learner:

1. **Meta-learner classifies failures** → Returns `failure_analysis`
2. **Mutation proposer uses failure_analysis** → Returns candidates
3. **Evaluator tests candidates** → Returns metrics
4. **Prompt manager stores best version** → Updates current prompt

See `meta_learner/` and `evaluation/` for implementation.

---

## Testing the Prompt System

Run the comprehensive test suite:

```bash
python prompts/test_prompt_system.py
```

This verifies:
- ✓ Base prompt creation and validation
- ✓ Persistence and versioning
- ✓ All mutation operators
- ✓ Failure-driven proposal
- ✓ Version creation and history
- ✓ Prompt rendering for LLM

---

## Next Steps

Phase 3 will implement the **LLM Inference Wrapper**, which:
- Reads the current prompt via `PromptSchema.to_prompt_string()`
- Sends it to the frozen LLM
- Parses structured decisions
- Logs all interactions

This completes the loop: Prompt → LLM → Agent → Evaluation → Mutation → Prompt
