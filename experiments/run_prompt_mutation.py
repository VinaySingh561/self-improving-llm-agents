# experiments/run_prompt_mutation.py

from agent.local_llm_client import LocalLLMClient
from meta_learner.structured_mutator import StructuredPromptMutator
from experiments.run_llm_frozen_baseline import run_llm_frozen


def summarize_failures(results):
    """
    Explicit, paper-safe failure summary.
    """
    summary = {"low_score": 0, "timeout": 0}

    for r in results:
        if r["score"] < 10:
            summary["low_score"] += 1
        if r["terminal_reason"] == "timeout":
            summary["timeout"] += 1

    return summary


def run_prompt_mutation(seed=42, episodes=10):
    print("=" * 60)
    print("PROMPT MUTATION EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------
    # Shared frozen LLM (CRITICAL)
    # --------------------------------------------------
    llm_client = LocalLLMClient(
        model_path="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        temperature=0.0,
        max_tokens=16,
    )

    # --------------------------------------------------
    # 1. Baseline evaluation
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
    # 2. Failure summarization
    # --------------------------------------------------
    failure_summary = summarize_failures(baseline_results)
    print("Failure summary:", failure_summary)

    # --------------------------------------------------
    # 3. Structured prompt mutation
    # --------------------------------------------------
    with open("prompts/v0_frozen.txt") as f:
        base_prompt = f.read()

    mutator = StructuredPromptMutator(seed=seed)
    candidate_prompt = mutator.mutate(
        base_prompt=base_prompt,
        failure_summary=failure_summary,
    )

    with open("prompts/v1_candidate.txt", "w") as f:
        f.write(candidate_prompt)

    print("\n[2] Candidate prompt written to prompts/v1_candidate.txt")

    # --------------------------------------------------
    # 4. Mutated prompt evaluation (SAME runner)
    # --------------------------------------------------
    print("\n[3] Evaluating mutated prompt")

    mutated_results = run_llm_frozen(
        seed=seed,
        episodes=episodes,
        llm_client=llm_client,
        prompt_path="prompts/v1_candidate.txt",
    )

    mutated_scores = [r["score"] for r in mutated_results]
    mutated_mean = sum(mutated_scores) / len(mutated_scores)

    print(f"Mutated mean score: {mutated_mean:.2f}")

    # --------------------------------------------------
    # 5. Accept / reject
    # --------------------------------------------------
    print("\n[4] Decision")
    if mutated_mean > baseline_mean:
        print("✓ Mutation ACCEPTED")
    else:
        print("✗ Mutation REJECTED")

    import json
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)

    with open(f"results/prompt_mutation_seed{seed}.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "mutated": mutated_results,
            "baseline_mean": baseline_mean,
            "mutated_mean": mutated_mean,
            "failure_summary": failure_summary,
        }, f, indent=2)



if __name__ == "__main__":
    run_prompt_mutation(seed=42, episodes=10)
