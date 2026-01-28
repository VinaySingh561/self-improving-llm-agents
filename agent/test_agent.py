"""Test suite for NetHack agent implementation."""

import pytest
import tempfile
from datetime import datetime

from agent.nethack_agent import (
    NetHackAgent,
    AgentStateManager,
    AgentState,
    DecisionRecord,
    EpisodeLog,
)
from agent.llm_wrapper import LLMConfig
from prompts import PromptManager
from skills import SkillLibrary


@pytest.fixture
def temp_dir():
    """Create temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def llm_config():
    """Create LLM config."""
    return LLMConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device="cpu",
    )


@pytest.fixture
def prompt_manager(temp_dir):
    """Create prompt manager with base prompt."""
    from prompts.prompt_schema import create_base_prompt
    manager = PromptManager(prompt_dir=temp_dir)
    base_prompt = create_base_prompt()
    manager.set_current_prompt(base_prompt)
    return manager


@pytest.fixture
def skill_library():
    """Create skill library."""
    return SkillLibrary()


@pytest.fixture
def agent(llm_config, prompt_manager, skill_library, temp_dir):
    """Create NetHack agent."""
    return NetHackAgent(
        llm_config=llm_config,
        prompt_manager=prompt_manager,
        skill_library=skill_library,
        log_dir=temp_dir,
    )


class TestAgentState:
    """Test AgentState dataclass."""
    
    def test_agent_state_creation(self):
        """Test creating agent state."""
        state = AgentState(
            episode_id="ep1",
            step=1,
            health=100,
            hunger=50,
            gold=0,
            level=1,
            score=0,
        )
        
        assert state.episode_id == "ep1"
        assert state.step == 1
        assert state.health == 100
        assert state.dead is False


class TestDecisionRecord:
    """Test DecisionRecord dataclass."""
    
    def test_decision_record_creation(self):
        """Test creating decision record."""
        record = DecisionRecord(
            episode_id="ep1",
            step=1,
            timestamp=datetime.now().isoformat(),
            agent_state={"health": 100},
            llm_prompt="Test prompt",
            llm_output={"skill": "explore"},
            llm_success=True,
            requested_skill="explore",
            available_skills=["explore", "fight"],
            executed_skill="explore",
            skill_status="success",
            keystrokes=['h'],
        )
        
        assert record.episode_id == "ep1"
        assert record.requested_skill == "explore"
        assert record.executed_skill == "explore"


class TestEpisodeLog:
    """Test EpisodeLog dataclass."""
    
    def test_episode_log_creation(self):
        """Test creating episode log."""
        log = EpisodeLog(
            episode_id="ep1",
            prompt_version="v1.0.0",
            start_time=datetime.now().isoformat(),
        )
        
        assert log.episode_id == "ep1"
        assert log.prompt_version == "v1.0.0"
        assert log.total_steps == 0
        assert len(log.decisions) == 0


class TestNetHackAgent:
    """Test NetHack agent."""
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_wrapper is not None
        assert agent.prompt_manager is not None
        assert agent.skill_library is not None
    
    def test_agent_start_episode(self, agent):
        """Test starting episode."""
        agent.start_episode("ep1", "v1.0.0")
        
        assert agent.current_episode is not None
        assert agent.current_episode.episode_id == "ep1"
        assert agent.current_episode.prompt_version == "v1.0.0"
        assert agent.decision_count == 0
    
    def test_agent_end_episode(self, agent):
        """Test ending episode."""
        agent.start_episode("ep1", "v1.0.0")
        log = agent.end_episode("test_reason")
        
        assert log.episode_id == "ep1"
        assert log.terminal_reason == "test_reason"
        assert log.end_time is not None
        assert agent.current_episode is None
    
    def test_agent_decide_and_act(self, agent):
        """Test decision and action execution."""
        agent.start_episode("ep1", "v1.0.0")
        
        observation = {
            'dead': False,
            'stuck': False,
            'visible_enemies': [],
            'health': 100,
            'max_health': 100,
            'inventory': {'items': []},
            'hunger': 50,
        }
        
        executed_skill, keystrokes = agent.decide_and_act(observation)
        
        assert executed_skill in ["explore", "fight", "flee", "eat", "use_item", "noop"]
        assert isinstance(keystrokes, list)
        assert agent.current_episode.total_steps == 1
    
    def test_agent_multiple_decisions(self, agent):
        """Test multiple decisions in episode."""
        agent.start_episode("ep1", "v1.0.0")
        
        observation = {
            'dead': False,
            'stuck': False,
            'visible_enemies': [],
            'health': 100,
            'max_health': 100,
            'inventory': {'items': []},
            'hunger': 50,
        }
        
        for i in range(5):
            agent.decide_and_act(observation)
        
        assert len(agent.current_episode.decisions) == 5
        assert agent.current_episode.total_steps == 5
    
    def test_agent_statistics_computation(self, agent):
        """Test statistics computation after episode."""
        agent.start_episode("ep1", "v1.0.0")
        
        observation = {
            'dead': False,
            'stuck': False,
            'visible_enemies': [],
            'health': 100,
            'max_health': 100,
            'inventory': {'items': []},
            'hunger': 50,
        }
        
        # Make several decisions
        for _ in range(3):
            agent.decide_and_act(observation)
        
        log = agent.end_episode("normal")
        
        assert log.llm_success_rate > 0
        assert len(log.skills_used) > 0
        assert log.avg_tokens_per_step >= 0
    
    def test_agent_fallback_skill_execution(self, agent):
        """Test fallback when requested skill unavailable."""
        agent.start_episode("ep1", "v1.0.0")
        
        # Dead observation - most skills unavailable
        observation = {
            'dead': True,
            'stuck': True,
            'visible_enemies': [],
            'health': 0,
            'max_health': 100,
            'inventory': {'items': []},
            'hunger': 50,
        }
        
        executed_skill, keystrokes = agent.decide_and_act(observation)
        
        assert executed_skill == "noop"
        assert keystrokes == []


class TestAgentStateManager:
    """Test agent state manager."""
    
    def test_save_and_load_episode_log(self, temp_dir):
        """Test saving and loading episode log."""
        log = EpisodeLog(
            episode_id="ep1",
            prompt_version="v1.0.0",
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            total_steps=5,
            terminal_reason="test_end",
        )
        
        path = f"{temp_dir}/episode.json"
        AgentStateManager.save_episode_log(log, path)
        
        loaded_log = AgentStateManager.load_episode_log(path)
        
        assert loaded_log.episode_id == "ep1"
        assert loaded_log.prompt_version == "v1.0.0"
        assert loaded_log.total_steps == 5
        assert loaded_log.terminal_reason == "test_end"
    
    def test_save_and_load_with_decisions(self, temp_dir):
        """Test saving and loading with decision records."""
        log = EpisodeLog(
            episode_id="ep1",
            prompt_version="v1.0.0",
            start_time=datetime.now().isoformat(),
        )
        
        decision = DecisionRecord(
            episode_id="ep1",
            step=1,
            timestamp=datetime.now().isoformat(),
            agent_state={"health": 100},
            llm_prompt="Test prompt",
            llm_output={"skill": "explore"},
            llm_success=True,
            requested_skill="explore",
            available_skills=["explore"],
            executed_skill="explore",
            skill_status="success",
            keystrokes=['h'],
        )
        
        log.decisions.append(decision)
        
        path = f"{temp_dir}/episode.json"
        AgentStateManager.save_episode_log(log, path)
        
        loaded_log = AgentStateManager.load_episode_log(path)
        
        assert len(loaded_log.decisions) == 1
        assert loaded_log.decisions[0].requested_skill == "explore"


class TestAgentIntegration:
    """Integration tests for full agent pipeline."""
    
    def test_full_episode_run(self, agent):
        """Test running a full episode."""
        agent.start_episode("ep_integration_1", "v1.0.0")
        
        observations = [
            {
                'dead': False, 'stuck': False,
                'visible_enemies': [], 'health': 100, 'max_health': 100,
                'inventory': {'items': []}, 'hunger': 50,
            },
            {
                'dead': False, 'stuck': False,
                'visible_enemies': [{'name': 'orc'}],
                'health': 80, 'max_health': 100, 'wielded_weapon': 'sword',
                'inventory': {'items': []}, 'hunger': 60,
            },
            {
                'dead': False, 'stuck': False,
                'visible_enemies': [], 'health': 50, 'max_health': 100,
                'inventory': {'items': [{'type': 'food', 'name': 'apple'}]},
                'hunger': 70,
            },
        ]
        
        skills_used = []
        for obs in observations:
            skill, keystrokes = agent.decide_and_act(obs)
            skills_used.append(skill)
        
        log = agent.end_episode("test_complete")
        
        assert log.total_steps == 3
        assert len(log.decisions) == 3
        assert len(skills_used) == 3
        assert log.llm_success_rate > 0
    
    def test_reproducibility_with_same_prompt(self, llm_config, prompt_manager, skill_library, temp_dir):
        """Test that same LLM decisions are made with same setup."""
        observation = {
            'dead': False, 'stuck': False,
            'visible_enemies': [{'name': 'orc'}], 'health': 80, 'max_health': 100,
            'wielded_weapon': 'sword',
            'inventory': {'items': []}, 'hunger': 50,
        }
        
        # Run two agents with same config
        agent1 = NetHackAgent(llm_config, prompt_manager, skill_library, temp_dir)
        agent2 = NetHackAgent(llm_config, prompt_manager, skill_library, temp_dir)
        
        agent1.start_episode("ep1", "v1.0.0")
        agent2.start_episode("ep2", "v1.0.0")
        
        skill1, keys1 = agent1.decide_and_act(observation)
        skill2, keys2 = agent2.decide_and_act(observation)
        
        # Both should make same skill decision with mock LLM
        assert skill1 == skill2
        # Note: keystrokes may differ for skills with internal state (e.g., explore pattern)
        assert isinstance(keys1, list) and isinstance(keys2, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
