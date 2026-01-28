import json
import os
from datetime import datetime

class PromptArchive:
    def __init__(self, archive_dir="prompts/archive"):
        self.archive_dir = archive_dir
        os.makedirs(self.archive_dir, exist_ok=True)

    def save(self, prompt_text: str, metrics: dict) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"prompt_{timestamp}.json"
        path = os.path.join(self.archive_dir, fname)

        with open(path, "w") as f:
            json.dump(
                {
                    "prompt": prompt_text,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        return path
