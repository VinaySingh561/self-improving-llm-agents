"""Test NLE integration."""

import pytest
import logging

logging.basicConfig(level=logging.INFO)


def test_nle_import():
    """Test that NLE can be imported."""
    try:
        import nle
        assert nle is not None
        print("✓ NLE library available")
    except ImportError:
        pytest.skip("NLE not installed")


def test_nle_wrapper_initialization():
    """Test NLE wrapper can be initialized."""
    try:
        from env.nle_wrapper import NLEWrapper
        env = NLEWrapper(seed=42, max_steps=100)
        assert env is not None
        assert env.seed == 42
        print("✓ NLE wrapper initialized")
        env.close()
    except ImportError:
        pytest.skip("NLE not installed")


def test_nle_reset():
    """Test NLE environment can be reset."""
    try:
        from env.nle_wrapper import NLEWrapper
        env = NLEWrapper(seed=42, max_steps=100)
        
        obs = env.reset()
        assert obs is not None
        assert "message" in obs
        assert "hp" in obs
        assert "score" in obs
        print("✓ NLE environment reset successfully")
        print(f"  Initial HP: {obs['hp']}/{obs['max_hp']}")
        print(f"  Initial Score: {obs['score']}")
        
        env.close()
    except ImportError:
        pytest.skip("NLE not installed")


def test_nle_step():
    """Test NLE environment step."""
    try:
        from env.nle_wrapper import NLEWrapper
        env = NLEWrapper(seed=42, max_steps=100)
        
        obs = env.reset()
        initial_hp = obs["hp"]
        
        # Take a step (rest action)
        obs, reward, done, info = env.step(".")
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        print("✓ NLE environment step works")
        print(f"  Action: rest")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        env.close()
    except ImportError:
        pytest.skip("NLE not installed")


def test_observation_parser():
    """Test observation parser."""
    try:
        from env.nle_wrapper import NLEWrapper, ObservationParser
        env = NLEWrapper(seed=42, max_steps=100)
        
        obs = env.reset()
        
        # Parse observation
        readable = ObservationParser.parse(obs)
        assert isinstance(readable, str)
        assert "Health:" in readable
        assert "Score:" in readable
        print("✓ Observation parser works")
        print(f"  Parsed output:\n{readable}")
        
        # Test health checks
        is_critical = ObservationParser.is_critical_health(obs)
        is_hungry = ObservationParser.is_hungry(obs)
        print(f"  Critical health: {is_critical}")
        print(f"  Is hungry: {is_hungry}")
        
        env.close()
    except ImportError:
        pytest.skip("NLE not installed")


def test_nle_observation_conversion():
    """Test NLE observation conversion to agent format."""
    try:
        from env.nle_wrapper import NLEWrapper
        from agent.nethack_agent import convert_nle_observation
        
        env = NLEWrapper(seed=42, max_steps=100)
        nle_obs = env.reset()
        
        # Convert to agent format
        agent_obs = convert_nle_observation(nle_obs)
        
        assert agent_obs is not None
        assert "health" in agent_obs
        assert "hunger" in agent_obs
        assert "score" in agent_obs
        assert "level" in agent_obs
        print("✓ NLE observation conversion works")
        print(f"  Agent observation: {agent_obs}")
        
        env.close()
    except ImportError:
        pytest.skip("NLE not installed")


def test_agent_with_nle():
    """Test agent can work with NLE observations."""
    try:
        from env.nle_wrapper import NLEWrapper
        from agent.llm_wrapper import LLMConfig
        from agent.nethack_agent import NetHackAgent
        from skills import SkillLibrary
        from prompts import PromptManager
        
        # Setup
        env = NLEWrapper(seed=42, max_steps=100)
        llm_config = LLMConfig(device="cpu", quantization_bits=4)
        prompt_manager = PromptManager()
        skill_library = SkillLibrary()
        
        agent = NetHackAgent(
            llm_config=llm_config,
            prompt_manager=prompt_manager,
            skill_library=skill_library,
        )
        
        # Start episode
        agent.start_episode("test_nle", "v1.0.0")
        
        # Get observation
        obs = env.reset()
        
        # Agent should handle NLE observation
        skill_name, keystrokes = agent.decide_and_act(obs)
        
        assert skill_name is not None
        assert isinstance(keystrokes, list)
        print("✓ Agent works with NLE observations")
        print(f"  Skill chosen: {skill_name}")
        print(f"  Keystrokes: {keystrokes}")
        
        agent.end_episode("test_complete")
        env.close()
    except ImportError as e:
        pytest.skip(f"NLE or dependencies not installed: {e}")


if __name__ == "__main__":
    print("Testing NLE Integration\n" + "="*50)
    
    test_nle_import()
    test_nle_wrapper_initialization()
    test_nle_reset()
    test_nle_step()
    test_observation_parser()
    test_nle_observation_conversion()
    test_agent_with_nle()
    
    print("\n" + "="*50)
    print("✓ All NLE integration tests passed!")
