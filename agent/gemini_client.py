# agent/gemini_client.py

import os
from google import genai
from google.genai.types import GenerateContentConfig


class GeminiDeterministicClient:
    """
    Deterministic Gemini wrapper (robust to empty responses).
    """

    def __init__(self, model_name="models/gemini-flash-latest"):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise RuntimeError("Set GEMINI_API_KEY environment variable")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        self.config = GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_output_tokens=5,
        )

    def query(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config,
        )

        # ---- ROBUST EXTRACTION ----
        if response.text:
            return response.text.strip()

        # Fallback: extract from candidates
        if response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            return part.text.strip()

        # Absolute fallback (safe action)
        return "."
