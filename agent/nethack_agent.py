"""NetHack agent: Orchestrates LLM decisions and skill execution.

Agent Loop:
1. Observe: Get current game state (observation)
2. Prompt LLM: Query LLM with prompt + observation
3. Extract: Parse LLM output to get skill decision
4. Check: Verify skill is executable in current state
5. Execute: Run skill and collect keystrokes
6. Step: Send keystrokes to environment
7. Record: Log decision, result, and metrics

Design:
- Deterministic: Same prompt + observation → same action
- Reproducible: Full audit trail of all decisions
- Stateless decision-making: LLM has no memory between calls
- Graceful fallback: If preferred skill unavailable, try alternatives
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from agent.llm_wrapper import LLMWrapper, LLMConfig
from skills import SkillLibrary, SkillExecutionStatus
from prompts import PromptSchema, PromptManager

logger = logging.getLogger(__name__)


def convert_nle_observation(nle_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert NLE observation to agent format.
    
    NLE observations have:
    - message: Game message
    - score: Current score
    - hp/max_hp: Health
    - satiation: Hunger level
    - dungeon_level: Current depth
    
    Agent format expects:
    - health, hunger, gold, level, score, dead, stuck
    """
    # Extract from NLE format
    hp = nle_obs.get("hp", 0)
    max_hp = nle_obs.get("max_hp", 100)
    satiation = nle_obs.get("satiation", 0)
    message = nle_obs.get("message", "").lower()
    
    # Convert to agent format
    return {
        "health": hp,
        "max_health": max_hp,
        "hunger": 100 if satiation > 1000 else (50 if satiation > 0 else 0),
        "gold": 0,  # Not available in basic NLE
        "level": nle_obs.get("dungeon_level", 1),
        "score": nle_obs.get("score", 0),
        "dead": not nle_obs.get("is_alive", True),
        "stuck": "not sure what to do" in message or "what do you want" in message,
        "message": nle_obs.get("message", ""),
        "exp": nle_obs.get("exp", 0),
    }


@dataclass
class DecisionRecord:
    """Record of a single agent decision."""
    episode_id: str
    step: int
    timestamp: str
    agent_state: Dict[str, Any]
    llm_prompt: str
    llm_output: Dict[str, Any]
    llm_success: bool
    requested_skill: str
    available_skills: List[str]
    executed_skill: str
    skill_status: str
    keystrokes: List[str]
    
    # Optional fields (with defaults must come last)
    llm_error: Optional[str] = None
    token_count: Optional[int] = None
    execution_time_ms: Optional[float] = None


@dataclass
class AgentState:
    """Current agent state within an episode."""
    episode_id: str
    step: int
    health: int
    hunger: int
    gold: int
    level: int
    score: int
    dead: bool = False
    stuck: bool = False


@dataclass
class EpisodeLog:
    """Log of an entire episode."""
    episode_id: str
    prompt_version: str
    start_time: str
    end_time: Optional[str] = None
    
    total_steps: int = 0
    terminal_reason: Optional[str] = None
    
    decisions: List[DecisionRecord] = field(default_factory=list)
    
    # Aggregated statistics
    skills_used: Dict[str, int] = field(default_factory=dict)
    skill_success_rates: Dict[str, float] = field(default_factory=dict)
    llm_success_rate: float = 0.0
    avg_tokens_per_step: float = 0.0


class NetHackAgent:
    """Agent for playing NetHack using LLM policy.
    
    Design:
    - LLM is called at skill boundaries only
    - All decisions are cached
    - State is fully serializable
    - Never outputs raw keystrokes
    """
    
    def __init__(
        self,
        llm_config: LLMConfig,
        prompt_manager: PromptManager,
        skill_library: SkillLibrary,
        log_dir: str = "./agent_logs",
    ):
        """
        Initialize agent.
        
        Args:
            llm_config: Configuration for LLM wrapper
            prompt_manager: Manager for prompt versioning
            skill_library: Library of available skills
            log_dir: Directory for logging
        """
        self.llm_wrapper = LLMWrapper(config=llm_config, log_dir=log_dir)
        self.prompt_manager = prompt_manager
        self.skill_library = skill_library
        self.log_dir = log_dir
        
        # Current episode tracking
        self.current_episode: Optional[EpisodeLog] = None
        self.decision_count = 0
        
        logger.info(
            "NetHackAgent initialized with LLM config: "
            f"model={llm_config.model_name}, device={llm_config.device}"
        )
    
    def start_episode(self, episode_id: str, prompt_version: str = "v1.0.0") -> None:
        """Start a new episode."""
        self.current_episode = EpisodeLog(
            episode_id=episode_id,
            prompt_version=prompt_version,
            start_time=datetime.now().isoformat(),
        )
        self.decision_count = 0
        logger.info(f"Started episode {episode_id} with prompt {prompt_version}")
    
    def end_episode(self, terminal_reason: str) -> EpisodeLog:
        """End current episode and return log."""
        if self.current_episode is None:
            raise RuntimeError("No active episode")
        
        self.current_episode.end_time = datetime.now().isoformat()
        self.current_episode.terminal_reason = terminal_reason
        
        # Compute aggregated statistics
        self._compute_episode_statistics()
        
        episode_log = self.current_episode
        self.current_episode = None
        
        logger.info(
            f"Ended episode {episode_log.episode_id}: "
            f"steps={episode_log.total_steps}, reason={terminal_reason}"
        )
        
        return episode_log
    
    def decide_and_act(self, observation: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Make a decision and execute action.
        
        Args:
            observation: Current game state (can be from NLE or mock)
            
        Returns:
            Tuple of (executed_skill_name, keystrokes)
        """
        if self.current_episode is None:
            raise RuntimeError("No active episode")
        
        # Convert NLE observation to agent format if needed
        if "is_alive" in observation:  # NLE format
            observation = convert_nle_observation(observation)
        
        self.decision_count += 1
        step = self.decision_count
        
        # 1. Extract agent state
        agent_state = self._extract_agent_state(observation)
        
        # 2. Get available skills
        available_skills = self.skill_library.list_available_skills(observation)
        
        # 3. Query LLM for skill decision
        prompt = self._build_prompt(agent_state, available_skills)
        
        llm_result = self.llm_wrapper.call(
            prompt=prompt,
            output_format="skill_decision",
        )
        
        requested_skill, llm_success, llm_error = self._extract_skill_decision(
            llm_result, available_skills
        )
        
        # 4. Execute skill (with fallback)
        executed_skill, keystrokes, skill_status = self._execute_skill(
            requested_skill, observation, available_skills
        )
        
        # 5. Record decision
        decision = DecisionRecord(
            episode_id=self.current_episode.episode_id,
            step=step,
            timestamp=datetime.now().isoformat(),
            agent_state=asdict(agent_state),
            llm_prompt=prompt,
            llm_output=llm_result.get('output', {}),
            llm_success=llm_success,
            llm_error=llm_error,
            requested_skill=requested_skill,
            available_skills=available_skills,
            executed_skill=executed_skill,
            skill_status=skill_status.value,
            keystrokes=keystrokes,
            token_count=llm_result.get('metadata', {}).get('tokens_used'),
        )
        
        self.current_episode.decisions.append(decision)
        self.current_episode.total_steps = step
        
        logger.debug(
            f"Step {step}: requested={requested_skill}, "
            f"executed={executed_skill}, status={skill_status.value}"
        )
        
        return executed_skill, keystrokes
    
    def _extract_agent_state(self, observation: Dict[str, Any]) -> AgentState:
        """Extract structured agent state from observation."""
        return AgentState(
            episode_id=self.current_episode.episode_id,
            step=self.decision_count,
            health=observation.get('health', 0),
            hunger=observation.get('hunger', 100),
            gold=observation.get('gold', 0),
            level=observation.get('level', 1),
            score=observation.get('score', 0),
            dead=observation.get('dead', False),
            stuck=observation.get('stuck', False),
        )
    
    def _build_prompt(
        self,
        agent_state: AgentState,
        available_skills: List[str]
    ) -> str:
        """Build prompt for LLM decision."""
        current_prompt = self.prompt_manager.get_current_prompt()
        base_prompt = current_prompt.to_prompt_string()
        
        state_summary = (
            f"Current State:\n"
            f"  Health: {agent_state.health}\n"
            f"  Hunger: {agent_state.hunger}\n"
            f"  Gold: {agent_state.gold}\n"
            f"  Level: {agent_state.level}\n"
            f"  Score: {agent_state.score}\n"
            f"  Available Skills: {', '.join(available_skills)}\n"
        )
        
        decision_prompt = (
            f"{base_prompt}\n\n"
            f"{state_summary}\n"
            f"Decide which skill to use (respond with JSON: {{'skill': '<skill_name>'}})"
        )
        
        return decision_prompt
    
    def _extract_skill_decision(
        self,
        llm_result: Dict[str, Any],
        available_skills: List[str]
    ) -> Tuple[str, bool, Optional[str]]:
        """Extract skill decision from LLM output."""
        if not llm_result.get('success', False):
            logger.warning(f"LLM call failed: {llm_result.get('error')}")
            default_skill = available_skills[0] if available_skills else "explore"
            return default_skill, False, llm_result.get('error')
        
        output = llm_result.get('output', {})
        skill = output.get('skill', 'explore')
        
        # Validate skill is in available list
        if skill not in available_skills:
            fallback = available_skills[0] if available_skills else skill
            logger.warning(
                f"LLM requested unavailable skill {skill}, "
                f"defaulting to {fallback}"
            )
            return skill, True, None  # LLM call succeeded, but decision invalid
        
        return skill, True, None
    
    def _execute_skill(
        self,
        requested_skill: str,
        observation: Dict[str, Any],
        available_skills: List[str]
    ) -> Tuple[str, List[str], SkillExecutionStatus]:
        """Execute requested skill with fallback."""
        # Try requested skill first
        if requested_skill in available_skills:
            result = self.skill_library.execute_skill(requested_skill, observation)
            
            if result.status in (
                SkillExecutionStatus.SUCCESS,
                SkillExecutionStatus.PARTIAL_SUCCESS
            ):
                return requested_skill, result.keystrokes, result.status
        
        # Fallback: try first available skill
        for skill_name in available_skills:
            result = self.skill_library.execute_skill(skill_name, observation)
            
            if result.status in (
                SkillExecutionStatus.SUCCESS,
                SkillExecutionStatus.PARTIAL_SUCCESS
            ):
                logger.info(
                    f"Requested skill {requested_skill} unavailable, "
                    f"fell back to {skill_name}"
                )
                return skill_name, result.keystrokes, result.status
        
        # All skills failed: return empty keystroke with failure status
        logger.warning("No skills could execute, sending no-op action")
        return "noop", [], SkillExecutionStatus.IMPOSSIBLE
    
    def _compute_episode_statistics(self) -> None:
        """Compute aggregated statistics for episode."""
        if self.current_episode is None:
            return
        
        episode = self.current_episode
        
        # Skill usage
        for decision in episode.decisions:
            skill = decision.executed_skill
            if skill not in episode.skills_used:
                episode.skills_used[skill] = 0
            episode.skills_used[skill] += 1
        
        # Skill success rates
        for skill_name in self.skill_library.list_skills():
            successes = 0
            executions = 0
            for decision in episode.decisions:
                if decision.executed_skill == skill_name:
                    executions += 1
                    if decision.skill_status in ('success', 'partial'):
                        successes += 1
            
            if executions > 0:
                episode.skill_success_rates[skill_name] = successes / executions
        
        # LLM success rate
        llm_successes = sum(1 for d in episode.decisions if d.llm_success)
        episode.llm_success_rate = (
            llm_successes / len(episode.decisions)
            if episode.decisions
            else 0.0
        )
        
        # Average tokens per step
        token_counts = [
            d.token_count for d in episode.decisions
            if d.token_count is not None
        ]
        episode.avg_tokens_per_step = (
            sum(token_counts) / len(token_counts)
            if token_counts
            else 0.0
        )


class AgentStateManager:
    """Save/load agent logs and replay episodes."""
    
    @staticmethod
    def save_episode_log(log: EpisodeLog, path: str) -> None:
        """Save episode log to JSON file."""
        data = asdict(log)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved episode log to {path}")
    
    @staticmethod
    def load_episode_log(path: str) -> EpisodeLog:
        """Load episode log from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct decisions as DecisionRecord objects
        decisions_data = data.pop('decisions', [])
        decisions = [
            DecisionRecord(
                **{k: v for k, v in d.items()
                   if k in DecisionRecord.__dataclass_fields__}
            )
            for d in decisions_data
        ]
        
        log = EpisodeLog(**{k: v for k, v in data.items()
                           if k in EpisodeLog.__dataclass_fields__})
        log.decisions = decisions
        
        return log

