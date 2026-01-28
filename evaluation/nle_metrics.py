from typing import List, Dict, Any
import numpy as np
from agent.episode_log import EpisodeLog


class NLEMetrics:
    """
    Compute NetHack-correct metrics from EpisodeLogs.
    """

    @staticmethod
    def per_seed_metrics(log: EpisodeLog) -> Dict[str, Any]:
        """Metrics for a single seed."""
        return {
            "seed": log.seed,
            "steps_survived": log.total_steps,
            "final_score": log.final_score,
            "final_dungeon_level": log.final_dungeon_level,
            "terminated_by": log.terminated_by,
            "died": log.terminated_by == "death",
        }

    @staticmethod
    def aggregate(logs: List[EpisodeLog]) -> Dict[str, Any]:
        """Aggregate metrics across seeds."""
        steps = np.array([l.total_steps for l in logs])
        depths = np.array([l.final_dungeon_level for l in logs])
        scores = np.array([l.final_score for l in logs])
        deaths = np.array([l.terminated_by == "death" for l in logs])

        return {
            "num_seeds": len(logs),

            # Primary metrics
            "steps_mean": float(np.mean(steps)),
            "steps_std": float(np.std(steps)),

            "depth_mean": float(np.mean(depths)),
            "depth_std": float(np.std(depths)),

            # Secondary metrics
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),

            "death_rate": float(np.mean(deaths)),
        }
