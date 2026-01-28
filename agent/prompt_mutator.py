class PromptMutator:
    """
    Generates controlled prompt mutations using the same frozen LLM.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def mutate(self, base_prompt: str, failure_summary: str) -> str:
        mutation_prompt = f"""
You are improving a NetHack-playing agent.

CURRENT PROMPT:
----------------
{base_prompt}
----------------

FAILURE ANALYSIS:
----------------
{failure_summary}
----------------

TASK:
Propose a NEW SYSTEM PROMPT that:
1. Avoids the observed failures
2. Remains simple and deterministic
3. Uses explicit rules, not randomness
4. Does NOT mention learning, rewards, or experiments

Return ONLY the new prompt text.
"""

        mutated = self.llm.query(mutation_prompt)
        return mutated.strip()
