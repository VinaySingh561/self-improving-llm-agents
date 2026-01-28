from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class EpisodeLog:
    episode_id: str
    seed: int
    start_time: str
    end_time: str

    total_steps: int
    final_score: int
    final_dungeon_level: int
    final_hp: int

    terminated_by: str  # "death", "timeout", "env_terminated"
    decisions: List[Dict[str, Any]] = field(default_factory=list)
