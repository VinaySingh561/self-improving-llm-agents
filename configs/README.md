# Configs Module

## Scientific Role
Provides deterministic configuration management with version tracking.

## Responsibilities
- Load and validate configuration files
- Provide defaults for all hyperparameters
- Track configuration versions
- Support config overrides for ablations

## Configuration Categories
- Environment (NLE seeds, episode length)
- LLM (model name, temperature, max_tokens)
- Agent (skill library, decision thresholds)
- Prompts (base prompt version, mutation operators)
- Meta-learner (failure detector thresholds, statistical test parameters)
- Evaluation (seed sets, metrics)
- Experiments (number of trials, baselines)

## Format
YAML files with schema validation.

## Status
Placeholder structure created. Implementation in progress.
