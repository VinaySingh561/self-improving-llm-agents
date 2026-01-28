"""
NLE wrapper for deterministic, seeded execution.

Integration with the actual NetHack Learning Environment (NLE).
Provides a minimal, publication-safe interface for real game interaction.
"""

from typing import Tuple, Dict, Any
import logging
import numpy as np

try:
    import gymnasium as gym
    import nle  # noqa: F401  (required to register envs)
    NLE_AVAILABLE = True
except ImportError:
    NLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class NLEWrapper:
    """
    Deterministic wrapper around NetHack Learning Environment.

    Design principles:
    - Real NLE interaction only
    - Seed is fixed per episode
    - Full ASCII action space (0–255)
    - Explicit episode horizon
    - Minimal observation parsing
    """

    # Full ASCII action space (standard NLE practice)
    ACTION_SPACE = list(range(256))

    # Safe single-step actions (used ONLY for tests / smoke runs)
    SAFE_TEST_ACTIONS = [
        ord("h"),  # left
        ord("j"),  # down
        ord("k"),  # up
        ord("l"),  # right
        ord("."),  # wait
    ]

    def __init__(self, seed: int, max_steps: int = 2000):
        if not NLE_AVAILABLE:
            raise ImportError("NLE not installed. Install with: pip install nle")

        self.seed = seed
        self.max_steps = max_steps
        self.env = None
        self.step_count = 0

        self._initialize_env()

    def _initialize_env(self) -> None:
        """Initialize NetHack environment."""
        try:
            self.env = gym.make("NetHackChallenge-v0")
            logger.info(f"NLE initialized (seed={self.seed})")
        except Exception as e:
            logger.error(f"Failed to initialize NLE: {e}")
            raise

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial parsed observation."""
        raw_obs, _ = self.env.reset(seed=self.seed)
        self.step_count = 0
        return self._parse_observation(raw_obs)

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute one NLE step.

        Args:
            action: int (ASCII code) or str (single character)

        Returns:
            parsed_obs, reward, done, info
        """
        try:
            # Convert action to ASCII code
            if isinstance(action, int):
                action_code = action
            elif isinstance(action, str) and len(action) == 1:
                action_code = ord(action)
            else:
                action_code = ord(".")  # safe no-op

            raw_obs, reward, terminated, truncated, info = self.env.step(action_code)

            self.step_count += 1
            timeout = self.step_count >= self.max_steps
            done = terminated or truncated or timeout

            parsed_obs = self._parse_observation(raw_obs)

            info = dict(info)
            info["step_count"] = self.step_count
            info["timeout"] = timeout
            info["action"] = action_code
            info["raw_obs"] = raw_obs
            return parsed_obs, float(reward), done, info

        except Exception as e:
            # NLE can throw internal errors for invalid command states.
            # We terminate the episode cleanly instead of crashing.
            logger.error(f"NLE step error (terminating episode): {e}")
            return (
                {},
                0.0,
                True,
                {
                    "error": str(e),
                    "step_count": self.step_count,
                    "timeout": False,
                    "terminated_by": "nle_error",
                },
            )

    def close(self) -> None:
        if self.env:
            self.env.close()

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------

    def _parse_observation(self, raw_obs: Dict) -> Dict[str, Any]:
        """
        Parse a minimal, defensible observation for agent use.

        Intentionally excludes:
        - map glyphs
        - inventory
        - enemy lists

        This enforces partial observability (explicitly stated in paper).
        """
        try:
            blstats = raw_obs.get("blstats", None)

            if blstats is not None and len(blstats) >= 7:
                score = int(blstats[2])
                dungeon_level = int(blstats[3])
                hp = int(blstats[4])
                max_hp = int(blstats[5])
                satiation = int(blstats[6])
            else:
                score = 0
                dungeon_level = 0
                hp = 0
                max_hp = 1
                satiation = 0

            message_raw = raw_obs.get("message", b"")
            if isinstance(message_raw, np.ndarray):
                message = bytes(message_raw).decode("utf-8", errors="ignore").strip()
            elif isinstance(message_raw, bytes):
                message = message_raw.decode("utf-8", errors="ignore").strip()
            else:
                message = str(message_raw)

            return {
                "message": message,
                "score": score,
                "dungeon_level": dungeon_level,
                "hp": hp,
                "max_hp": max_hp,
                "satiation": satiation,
                "is_alive": hp > 0,
            }

        except Exception as e:
            logger.warning(f"Observation parsing error: {e}")
            return {
                "message": "",
                "score": 0,
                "dungeon_level": 0,
                "hp": 0,
                "max_hp": 1,
                "satiation": 0,
                "is_alive": False,
            }


# ----------------------------------------------------------------------
# Observation-to-text adapter (LLM-facing)
# ----------------------------------------------------------------------

class ObservationParser:
    """Convert parsed observations into text prompts for LLM agents."""

    @staticmethod
    def parse(obs: Dict[str, Any]) -> str:
        lines = []

        if obs.get("message"):
            lines.append(f"Message: {obs['message']}")

        hp = obs["hp"]
        max_hp = obs["max_hp"]
        hp_pct = 100 * hp / max_hp if max_hp > 0 else 0
        lines.append(f"Health: {hp}/{max_hp} ({hp_pct:.0f}%)")

        satiation = obs["satiation"]
        if satiation > 1000:
            hunger = "Overfull"
        elif satiation > 150:
            hunger = "Not hungry"
        elif satiation > 0:
            hunger = "Hungry"
        elif satiation > -500:
            hunger = "Very hungry"
        else:
            hunger = "Famished"
        lines.append(f"Hunger: {hunger}")

        lines.append(f"Score: {obs['score']}")
        lines.append(f"Dungeon Level: {obs['dungeon_level']}")

        if not obs["is_alive"]:
            lines.append("STATUS: DEAD")

        return "\n".join(lines)
