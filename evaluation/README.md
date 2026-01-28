# Evaluation Module

## Scientific Role
Implements rigorous, reproducible evaluation with strict separation of training/validation/test sets.

## Responsibilities
- Maintain frozen seed sets (train/val/test)
- Execute agents deterministically
- Compute metrics (score, survival time, failure diversity)
- Log all evaluation runs
- Prevent data leakage

## Key Components
- `Evaluator`: Main evaluation harness
- `MetricsComputer`: Aggregate metrics
- `SeedManager`: Frozen seed management

## Metrics
- Average score per episode
- Survival time (turns survived)
- Failure diversity (number of distinct failure modes)
- Success rate (episodes reaching target)
- Confidence intervals (via bootstrap)

## Design Constraints
- No adaptive sampling
- Fixed batch sizes
- Deterministic order
- Full logging of trajectories
- No cherry-picking seeds

## Evaluation Modes
- Training evaluation (internal, on train seeds)
- Validation evaluation (for mutation selection)
- Test evaluation (final, frozen, reported)

## Status
Placeholder structure created. Implementation in progress.
