# Prompts Module

## Scientific Role
Manages machine-editable prompts designed for reproducible self-improvement via structured mutations.

## Responsibilities
- Define prompt schema (structured, not free-text)
- Decompose prompts into frozen and mutable parts
- Version all prompt changes
- Implement mutation operators
- Prevent invalid mutations

## Prompt Structure (YAML/JSON)
- Task definition (frozen)
- Rules and constraints (mostly frozen)
- Skill descriptions (frozen)
- Decision policy parameters (mutable)

## Key Components
- `PromptSchema`: JSON schema for valid prompts
- `PromptManager`: Versioning and storage
- `MutationOperator`: Structured mutation logic

## Mutation Types
- Adjust decision thresholds
- Modify skill priorities
- Update risk parameters
- Refine state interpretation rules

## Design Constraints
- No free-text rewriting
- All mutations must be version-tracked
- Mutations must be reversible (rollback support)
- Version history must be immutable

## Status
Placeholder structure created. Implementation in progress.
