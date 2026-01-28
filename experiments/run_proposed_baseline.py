from env.nle_wrapper import NLEWrapper
from agent.llm_static_agent import LLMStaticAgent

def run_proposed(seed: int, episodes: int = 10):
    env = NLEWrapper(seed=seed, max_steps=1000)
    agent = LLMStaticAgent(env.SAFE_TEST_ACTIONS)

    results = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            obs_text = obs["message"]
            action = agent.act(obs_text)
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
    SEED = 42
    EPISODES = 10

    print(f"Running proposed baseline (LLM agent) — seed={SEED}")
    results = run_proposed(seed=SEED, episodes=EPISODES)

    print("-" * 40)
    for r in results:
        print(r)

    scores = [r["score"] for r in results]
    alive = [r["alive"] for r in results]

    print("-" * 40)
    print(f"Episodes: {len(results)}")
    print(f"Mean score: {sum(scores)/len(scores):.2f}")
    print(f"Survival rate: {sum(alive)/len(alive):.2%}")
