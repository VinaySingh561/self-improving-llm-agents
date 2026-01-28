# agent/llm_frozen_agent.py

class LLMFrozenAgent:
    """
    Frozen LLM agent: same model, same decoding, only prompt varies.
    """

    def __init__(self, llm_client, prompt_path: str):
        self.llm = llm_client
        with open(prompt_path, "r") as f:
            self.base_prompt = f.read()

    def act(self, observation_text: str) -> int:
        full_prompt = (
            self.base_prompt
            + "\n\nOBSERVATION:\n"
            + observation_text
            + "\n\nACTION (single character):"
        )

        response = self.llm.query(full_prompt)

        if len(response) == 0:
            return ord(".")

        return ord(response[0])
