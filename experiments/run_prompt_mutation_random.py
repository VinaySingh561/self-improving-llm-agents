# experiments/run_prompt_mutation_random.py

import random
from agent.local_llm_client import LocalLLMClient
from experiments.run_llm_frozen_baseline import run_llm_frozen


def random_prompt_mutation(base_prompt: str, seed: int) -> str:
    """
    RANDOM prompt mutation baseline.

    This deliberately ignores failure summaries.
    The mutation is shallow and unguided:
    - random line deletion
    - random line duplication
    - random reordering

    This serves as a negative control in Ablation A1.
    """
    rng = random.Random(seed)

    lines = base_prompt.splitlines()
    if len(lines) < 4:
        return base_prompt  # safety

    mutation_type = rng.choice(["delete", "duplicate", "shuffle"])

    if mutation_type == "delete":
        idx = rng.randrange(len(lines))
        lines.pop(idx)

    elif mutation_type == "duplicate":
        idx = rng.randrange(len(lines))
        lines.insert(idx, lines[idx])

    elif mutation_type == "shuffle":
        rng.shuffle(lines)

    return "\n".join(lines)


def run_prompt_mutation_random(seed=42, episodes=10):
    print("=" * 60)
    print("PROMPT MUTATION EXPERIMENT (RANDOM BASELINE)")
    print("=" * 60)

    # --------------------------------------------------
    # Shared frozen LLM (identical to structured run)
    # --------------------------------------------------
    llm_client = LocalLLMClient(
        model_path="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        temperature=0.0,
        max_tokens=16,
    )

    # --------------------------------------------------
    # 1. Frozen baseline
    # --------------------------------------------------
    print("\n[1] Running baseline (frozen prompt)")
    baseline_results = run_llm_frozen(
        seed=seed,
        episodes=episodes,
        llm_client=llm_client,
        prompt_path="prompts/v0_frozen.txt",
    )

    baseline_scores = [r["score"] for r in baseline_results]
    baseline_mean = sum(baseline_scores) / len(baseline_scores)

    print(f"Baseline mean score: {baseline_mean:.2f}")

    # --------------------------------------------------
    # 2. Random prompt mutation
    # --------------------------------------------------
    with open("prompts/v0_frozen.txt") as f:
        base_prompt = f.read()

    candidate_prompt = random_prompt_mutation(base_prompt, seed)

    with open("prompts/v1_random_candidate.txt", "w") as f:
        f.write(candidate_prompt)

    print("\n[2] Random candidate prompt written to prompts/v1_random_candidate.txt")

    # --------------------------------------------------
    # 3. Evaluate mutated prompt
    # --------------------------------------------------
    print("\n[3] Evaluating random mutation")

    mutated_results = run_llm_frozen(
        seed=seed,
        episodes=episodes,
        llm_client=llm_client,
        prompt_path="prompts/v1_random_candidate.txt",
    )

    mutated_scores = [r["score"] for r in mutated_results]
    mutated_mean = sum(mutated_scores) / len(mutated_scores)

    print(f"Random mutation mean score: {mutated_mean:.2f}")

    # --------------------------------------------------
    # 4. Accept / reject (logged, not enforced)
    # --------------------------------------------------
    print("\n[4] Decision")
    if mutated_mean > baseline_mean:
        print("✓ Random mutation improved over baseline (rare)")
    else:
        print("✗ Random mutation did not improve baseline")

    # --------------------------------------------------
    # 5. Save results (paper-safe)
    # --------------------------------------------------
    import json
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)

    with open(f"results/prompt_mutation_random_seed{seed}.json", "w") as f:
        json.dump(
            {
                "baseline": baseline_results,
                "random_mutation": mutated_results,
                "baseline_mean": baseline_mean,
                "random_mutation_mean": mutated_mean,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run_prompt_mutation_random(seed=42, episodes=10)
