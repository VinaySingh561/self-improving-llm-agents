import json
import os
from typing import List, Dict
from datetime import datetime

from env.nle_wrapper import NLEWrapper
from experiments.run_real_episode import run_real_episode
from agent.episode_log import EpisodeLog


class NLEExperimentRunner:
    """
    Runs real NetHack experiments across multiple seeds.

    Design principles:
    - Seed is the unit of analysis
    - No learning during execution
    - Deterministic per seed
    - All results serialized
    """

    def __init__(
        self,
        agent,
        max_steps: int,
        output_dir: str = "results_nle",
    ):
        self.agent = agent
        self.max_steps = max_steps
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        seeds: List[int],
        tag: str,
    ) -> List[EpisodeLog]:
        """
        Run one episode per seed.

        Args:
            seeds: list of random seeds
            tag: identifier (e.g. static / random / proposed)

        Returns:
            List of EpisodeLog
        """

        all_logs: List[EpisodeLog] = []

        for seed in seeds:
            print(f"[NLE] Running seed={seed} | tag={tag}")

            env = NLEWrapper(seed=seed, max_steps=self.max_steps)

            episode_log = run_real_episode(
                agent=self.agent,
                env_wrapper=env,
                seed=seed,
                max_steps=self.max_steps,
            )

            env.close()
            all_logs.append(episode_log)

            self._save_episode_log(episode_log, tag)

        self._save_summary(all_logs, tag)
        return all_logs

    def _save_episode_log(self, log: EpisodeLog, tag: str):
        """Save one episode log as JSON."""
        path = os.path.join(
            self.output_dir,
            f"{tag}_seed{log.seed}.json",
        )

        with open(path, "w") as f:
            json.dump(log.__dict__, f, indent=2)

    def _save_summary(self, logs: List[EpisodeLog], tag: str):
        """Save lightweight run summary."""
        summary = {
            "tag": tag,
            "num_seeds": len(logs),
            "timestamp": datetime.now().isoformat(),
            "seeds": [log.seed for log in logs],
            "steps_survived": [log.total_steps for log in logs],
            "final_scores": [log.final_score for log in logs],
            "final_dungeon_levels": [log.final_dungeon_level for log in logs],
            "termination_causes": [log.terminated_by for log in logs],
        }

        path = os.path.join(self.output_dir, f"{tag}_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
