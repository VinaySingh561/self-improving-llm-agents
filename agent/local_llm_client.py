# agent/local_llm_client.py

from llama_cpp import Llama


class LocalLLMClient:
    """
    Deterministic local LLM wrapper (CPU).
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 16,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            logits_all=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n"],
        )

        return output["choices"][0]["text"].strip()
