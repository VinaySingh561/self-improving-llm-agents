import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import random
from env.nle_wrapper import NLEWrapper

def run_static(seed: int, episodes: int = 10):
    env = NLEWrapper(seed=seed, max_steps=1000)
    results = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            action = random.choice(env.SAFE_TEST_ACTIONS)
            obs, reward, done, info = env.step(action)
            steps += 1

        results.append({
            "seed": seed,
            "episode": ep,
            "steps": steps,
            "score": obs.get("score", 0),
            "alive": obs.get("is_alive", False),
            "terminal_reason": "timeout" if steps >= env.max_steps else "terminated",
        })


    env.close()
    return results


if __name__ == "__main__":
    results = run_static(seed=42)
    for r in results:
        print(r)
    print("✓ Observation parser module imported successfully")