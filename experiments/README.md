# Experiments Module

## Scientific Role
Orchestrates reproducible experimental comparisons with multiple random seeds and statistical rigor.

## Responsibilities
- Run baseline experiments (static prompt, random mutation)
- Run main self-improving agent
- Aggregate results across random seeds
- Compute confidence intervals
- Generate publication-ready plots and tables

## Baselines
- **Static Prompt**: No mutations, fixed policy
- **Random Mutation**: Random mutations at each cycle
- **Proposed Method**: Failure-driven mutation selection

## Experimental Design
- N_seeds = 5 (multiple random seeds)
- Episodes per eval = 20
- Mutations per cycle = 5
- Validation size = 10 seeds
- Test size = 20 seeds

## Metrics Computed
- Mean score ± 95% CI
- Mean survival time ± 95% CI
- Failure diversity ± 95% CI
- Success rate ± 95% CI

## Output Artifacts
- CSV files with raw metrics
- PNG plots (learning curves, comparison bars)
- LaTeX tables for paper

## Status
Placeholder structure created. Implementation in progress.
