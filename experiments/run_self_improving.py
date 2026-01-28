from env.nle_wrapper import NLEWrapper, ObservationParser
from agent.llm_frozen_agent import LLMFrozenAgent
from agent.prompt_mutator import PromptMutator
from agent.prompt_archive import PromptArchive
from agent.local_llm_client import LocalDeterministicLLM   # your llama.cpp client
from utils.failure_summary import summarize_failures

VAL_SEEDS = [123, 456]
TRAIN_SEEDS = [42]
EPISODES = 5


def evaluate_prompt(prompt_path, seeds):
    results = []

    llm = LocalDeterministicLLM(
        model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_threads=8,          # adjust to your CPU
    )
    agent = LLMFrozenAgent(llm_client=llm, prompt_path=prompt_path)

    for seed in seeds:
        env = NLEWrapper(seed=seed, max_steps=1000)

        for ep in range(EPISODES):
            obs = env.reset()
            done = False
            steps = 0

            while not done:
                obs_text = ObservationParser.parse(obs)
                action = agent.act(obs_text)
                obs, reward, done, info = env.step(action)
                steps += 1

            results.append(
                {
                    "seed": seed,
                    "episode": ep,
                    "steps": steps,
                    "score": obs["score"],
                    "alive": obs["is_alive"],
                }
            )

        env.close()

    mean_score = sum(r["score"] for r in results) / len(results)
    survival = sum(r["alive"] for r in results) / len(results)

    return mean_score, survival, results


def run_self_improving():
    llm = LocalDeterministicLLM(
        model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_threads=8,          # adjust to your CPU
    )
    mutator = PromptMutator(llm)
    archive = PromptArchive()

    base_prompt_path = "prompts/v0_frozen.txt"
    base_prompt = open(base_prompt_path).read()

    base_score, base_survival, logs = evaluate_prompt(
        base_prompt_path, VAL_SEEDS
    )

    print(f"BASELINE score={base_score:.2f}, survival={base_survival:.2%}")

    failure_summary = summarize_failures(logs)
    mutated_prompt = mutator.mutate(base_prompt, failure_summary)

    tmp_path = "prompts/tmp_candidate.txt"
    with open(tmp_path, "w") as f:
        f.write(mutated_prompt)

    new_score, new_survival, _ = evaluate_prompt(tmp_path, VAL_SEEDS)

    print(f"MUTATION score={new_score:.2f}, survival={new_survival:.2%}")

    if new_score > base_score and new_survival >= base_survival:
        archive.save(mutated_prompt, {
            "mean_score": new_score,
            "survival": new_survival,
        })
        print("✓ Mutation accepted")
    else:
        print("✗ Mutation rejected")


if __name__ == "__main__":
    run_self_improving()
