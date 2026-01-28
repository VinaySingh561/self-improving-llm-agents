import random

class LLMStaticAgent:
    """
    Deterministic reasoning-based agent.
    No learning, but state-aware.
    """

    def __init__(self, action_space):
        self.action_space = action_space
        self.wait = ord(".")
        self.moves = [ord(c) for c in ["h","j","k","l"]]

    def act(self, observation_text: str) -> int:
        # Very simple reasoning rules
        if "hungry" in observation_text.lower():
            return self.wait  # waiting avoids risk
        if "you see" in observation_text.lower():
            return random.choice(self.moves)
        return random.choice(self.action_space)

