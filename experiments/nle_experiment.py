"""Real NLE experiment: Runs agent against actual NetHack environment.

This replaces the mock experiment with real game play.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import json
from datetime import datetime

from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.nethack_agent import NetHackAgent
from skills import SkillLibrary
from prompts import PromptManager, PromptSchema
from meta_learner import FailureDetector
from env.nle_wrapper import NLEWrapper, ObservationParser

logger = logging.getLogger(__name__)


@dataclass
class NLEExperimentResult:
    """Result of a single NLE game."""
    episode_id: str
    prompt_version: str
    seed: int
    success: bool
    final_score: int
    steps_taken: int
    death_reason: str
    timestamp: str
    agent_log: Dict[str, Any]


class NLEExperimentRunner:
    """Runs experiments with real NetHack games."""
    
    def __init__(self, log_dir: str = "./nle_logs"):
        """Initialize experiment runner."""
        self.log_dir = log_dir
        self.llm_config = LLMConfig(
            model_name="mistral-7b-instruct",
            device="cpu",
            quantization_bits=4,
            temperature=0.0,
        )
        
    def run_episode(
        self,
        prompt_version: str = "v1.0.0",
        seed: int = 42,
        max_steps: int = 1000,
    ) -> NLEExperimentResult:
        """
        Run a single episode against real NetHack.
        
        Args:
            prompt_version: Prompt version to use
            seed: Random seed
            max_steps: Maximum steps per episode
            
        Returns:
            Experiment result
        """
        try:
            # Initialize components
            prompt_manager = PromptManager()
            skill_library = SkillLibrary()
            
            env = NLEWrapper(seed=seed, max_steps=max_steps)
            agent = NetHackAgent(
                llm_config=self.llm_config,
                prompt_manager=prompt_manager,
                skill_library=skill_library,
                log_dir=self.log_dir,
            )
            
            # Start episode
            episode_id = f"nle_real_{seed}_{datetime.now().isoformat()[:19].replace(':', '-')}"
            agent.start_episode(episode_id, prompt_version)
            
            # Initial observation
            obs = env.reset()
            logger.info(f"Episode {episode_id} started. Initial observation:")
            logger.info(ObservationParser.parse(obs))
            
            step = 0
            final_score = 0
            death_reason = "unknown"
            
            # Main game loop
            while step < max_steps:
                try:
                    # Agent decides and acts
                    skill_name, keystrokes = agent.decide_and_act(obs)
                    
                    logger.debug(f"Step {step}: skill={skill_name}, actions={keystrokes}")
                    
                    # Execute keystrokes in environment
                    for keystroke in keystrokes:
                        obs, reward, done, info = env.step(keystroke)
                        step += 1
                        
                        if done:
                            break
                    
                    # Check terminal conditions
                    if not obs.get("is_alive", True):
                        death_reason = "character_death"
                        break
                    
                    if info.get("timeout"):
                        death_reason = "timeout"
                        break
                    
                    final_score = obs.get("score", 0)
                    
                except Exception as e:
                    logger.error(f"Error during step {step}: {e}")
                    death_reason = f"error: {str(e)}"
                    break
            
            # End episode
            episode_log = agent.end_episode(death_reason)
            
            result = NLEExperimentResult(
                episode_id=episode_id,
                prompt_version=prompt_version,
                seed=seed,
                success=episode_log.total_steps > 10 and final_score > 0,
                final_score=final_score,
                steps_taken=step,
                death_reason=death_reason,
                timestamp=datetime.now().isoformat(),
                agent_log=agent.current_episode.to_dict() if agent.current_episode else {},
            )
            
            # Clean up
            env.close()
            
            logger.info(
                f"Episode {episode_id} finished: "
                f"score={final_score}, steps={step}, reason={death_reason}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fatal error in episode: {e}", exc_info=True)
            raise
    
    def run_multi_episode_experiment(
        self,
        prompt_version: str = "v1.0.0",
        num_episodes: int = 5,
        seeds: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple episodes and aggregate results.
        
        Args:
            prompt_version: Prompt version to use
            num_episodes: Number of episodes to run
            seeds: List of seeds to use
            
        Returns:
            Aggregated results
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 999][:num_episodes]
        
        results = []
        scores = []
        steps = []
        successes = 0
        
        for seed in seeds:
            try:
                result = self.run_episode(
                    prompt_version=prompt_version,
                    seed=seed,
                    max_steps=1000,
                )
                results.append(result)
                scores.append(result.final_score)
                steps.append(result.steps_taken)
                if result.success:
                    successes += 1
                    
            except Exception as e:
                logger.error(f"Failed to run episode with seed {seed}: {e}")
        
        # Compute aggregates
        import numpy as np
        scores_arr = np.array(scores)
        steps_arr = np.array(steps)
        
        return {
            "prompt_version": prompt_version,
            "num_episodes": len(results),
            "success_rate": successes / len(results) if results else 0,
            "avg_score": float(np.mean(scores_arr)) if len(scores) > 0 else 0,
            "std_score": float(np.std(scores_arr)) if len(scores) > 0 else 0,
            "min_score": float(np.min(scores_arr)) if len(scores) > 0 else 0,
            "max_score": float(np.max(scores_arr)) if len(scores) > 0 else 0,
            "avg_steps": float(np.mean(steps_arr)) if len(steps) > 0 else 0,
            "std_steps": float(np.std(steps_arr)) if len(steps) > 0 else 0,
            "results": [
                {
                    "episode_id": r.episode_id,
                    "score": r.final_score,
                    "steps": r.steps_taken,
                    "success": r.success,
                    "reason": r.death_reason,
                }
                for r in results
            ],
        }


if __name__ == "__main__":
    """Test real NLE integration."""
    logging.basicConfig(level=logging.INFO)
    
    runner = NLEExperimentRunner()
    
    # Run a single episode as test
    print("Running single NLE episode (real game)...")
    result = runner.run_episode(prompt_version="v1.0.0", seed=42, max_steps=100)
    
    print(f"\nEpisode Result:")
    print(f"  Score: {result.final_score}")
    print(f"  Steps: {result.steps_taken}")
    print(f"  Success: {result.success}")
    print(f"  Reason: {result.death_reason}")
