from typing import Callable
from datetime import datetime
from agent.episode_log import EpisodeLog


def run_real_episode(
    agent,
    env_wrapper,
    seed: int,
    max_steps: int,
) -> EpisodeLog:
    """
    Run a single real NetHack episode with fixed seed.

    Contract:
    - No learning during episode
    - Deterministic given seed
    - Real NLE interaction only
    """

    obs = env_wrapper.reset()
    start_time = datetime.now().isoformat()

    total_steps = 0
    terminated_by = "unknown"

    while True:
        # Agent chooses an action (single ASCII char or int)
        action = agent.act(obs)

        # Step environment
        next_obs, reward, done, info = env_wrapper.step(action)
        total_steps += 1

        # Log decision (minimal but auditable)
        agent.log_decision(
            observation=obs,
            action=action,
            info=info,
        )

        obs = next_obs

        # Termination logic (explicit and ordered)
        if not obs.get("is_alive", True):
            terminated_by = "death"
            break

        if info.get("timeout", False):
            terminated_by = "timeout"
            break

        if done:
            terminated_by = "env_terminated"
            break

        if total_steps >= max_steps:
            terminated_by = "timeout"
            break

    end_time = datetime.now().isoformat()

    return EpisodeLog(
        episode_id=f"seed_{seed}",
        seed=seed,
        start_time=start_time,
        end_time=end_time,
        total_steps=total_steps,
        final_score=obs.get("score", 0),
        final_dungeon_level=obs.get("dungeon_level", 0),
        final_hp=obs.get("hp", 0),
        terminated_by=terminated_by,
        decisions=agent.flush_decisions(),
    )
