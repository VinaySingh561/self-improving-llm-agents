# Agent Module

## Scientific Role
Orchestrates decision-making by calling the frozen LLM at skill boundaries, without outputting raw keystrokes.

## Responsibilities
- Call LLM only at skill transitions
- Cache and reuse decisions
- Track internal state (current skill, progress, failures)
- Execute skills deterministically
- Be fully interruptible and resumable

## Key Components
- `NetHackAgent`: Main agent class
- `AgentStateManager`: State tracking
- `SkillExecutor`: Clean skill interface

## Design Constraints
- Never output raw keystrokes directly
- All decisions flow through LLM → Skill → Keystroke chain
- State must be serializable for reproducibility

## Example Skills
- `explore`: Navigate unexplored areas
- `fight`: Engage enemies
- `flee`: Escape danger
- `eat`: Consume food
- `use_item`: Manipulate inventory

## Status
Placeholder structure created. Implementation in progress.
