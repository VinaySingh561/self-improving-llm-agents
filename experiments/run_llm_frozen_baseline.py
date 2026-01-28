# # experiments/run_llm_frozen_baseline.py

from env.nle_wrapper import NLEWrapper, ObservationParser
from agent.llm_frozen_agent import LLMFrozenAgent

def run_llm_frozen(seed, episodes, llm_client, prompt_path):
    env = NLEWrapper(seed=seed, max_steps=1000)

    agent = LLMFrozenAgent(
        llm_client=llm_client,
        prompt_path=prompt_path,
    )

    results = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            obs_text = ObservationParser.parse(obs)
            action = agent.act(obs_text)
            obs, _, done, info = env.step(action)
            steps += 1

        results.append({
            "seed": seed,
            "episode": ep,
            "steps": steps,
            "score": obs["score"],
            "alive": obs["is_alive"],
            "terminal_reason": "timeout" if info.get("timeout") else "terminated",
        })

    env.close()
    return results


# from collections import Counter

# def run_llm_frozen(seed, episodes, llm_client, prompt_path):
#     env = NLEWrapper(seed=seed, max_steps=1000)
#     agent = LLMFrozenAgent(
#         llm_client=llm_client,
#         prompt_path=prompt_path,
#     )

#     results = []
#     action_counter = Counter()

#     MOVE_ACTIONS = set("hjkl yubn")  # movement keys (include diagonals)

#     for ep in range(episodes):
#         obs = env.reset()
#         done = False
#         steps = 0

#         while not done:
#             obs_text = ObservationParser.parse(obs)
#             action = agent.act(obs_text)

#             # -------------------------------
#             # ACTION LOGGING (NEW)
#             # -------------------------------
#             if isinstance(action, int):
#                 action_char = chr(action)
#             else:
#                 action_char = action

#             action_counter[action_char] += 1

#             obs, _, done, info = env.step(action)
#             steps += 1

#         results.append(
#             {
#                 "seed": seed,
#                 "episode": ep,
#                 "steps": steps,
#                 "score": obs["score"],
#                 "alive": obs["is_alive"],
#                 "terminal_reason": "timeout" if info.get("timeout") else "terminated",
#             }
#         )

#     env.close()

#     # -----------------------------------
#     # PRINT ACTION DISTRIBUTION SUMMARY
#     # -----------------------------------
#     total_actions = sum(action_counter.values())
#     wait_count = action_counter["."]
#     move_count = sum(action_counter[a] for a in MOVE_ACTIONS)
#     other_count = total_actions - wait_count - move_count

#     print("\nACTION DISTRIBUTION")
#     print("-------------------")
#     print(f"Total actions: {total_actions}")
#     print(f"WAIT (.): {wait_count} ({wait_count/total_actions:.2%})")
#     print(f"MOVE: {move_count} ({move_count/total_actions:.2%})")
#     print(f"OTHER: {other_count} ({other_count/total_actions:.2%})")

#     return results


# if __name__ == "__main__":
#     from agent.local_llm_client import LocalLLMClient

#     llm_client = LocalLLMClient(
#         model_path="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
#         temperature=0.0,
#         max_tokens=16,
#     )

#     print("Running frozen LLM baseline")
#     print("-" * 40)

#     run_llm_frozen(
#         seed=42,
#         episodes=10,
#         llm_client=llm_client,
#         prompt_path="prompts/v0_frozen.txt",
#     )


# from env.nle_wrapper import NLEWrapper, ObservationParser
# from agent.llm_frozen_agent import LLMFrozenAgent
# from collections import Counter


# def run_llm_frozen(seed, episodes, llm_client, prompt_path):
#     env = NLEWrapper(seed=seed, max_steps=1000)

#     agent = LLMFrozenAgent(
#         llm_client=llm_client,
#         prompt_path=prompt_path,
#     )

#     results = []

#     # -------------------------------
#     # ACTION DISTRIBUTION TRACKER
#     # -------------------------------
#     action_counter = Counter()
#     MOVE_ACTIONS = set("hjkl yubn")  # 8-direction movement

#     for ep in range(episodes):
#         obs = env.reset()
#         done = False
#         steps = 0

#         while not done:
#             obs_text = ObservationParser.parse(obs)
#             action = agent.act(obs_text)

#             # -------- PASSIVE LOGGING --------
#             if isinstance(action, int):
#                 action_char = chr(action)
#             else:
#                 action_char = action

#             action_counter[action_char] += 1
#             # --------------------------------

#             obs, _, done, info = env.step(action)
#             steps += 1

#         results.append({
#             "seed": seed,
#             "episode": ep,
#             "steps": steps,
#             "score": obs["score"],
#             "alive": obs["is_alive"],
#             "terminal_reason": "timeout" if info.get("timeout") else "terminated",
#         })

#     env.close()

#     # ======================================
#     # PRINT ACTION DISTRIBUTION (END ONLY)
#     # ======================================
#     total = sum(action_counter.values())
#     wait = action_counter["."]
#     move = sum(action_counter[a] for a in MOVE_ACTIONS)
#     other = total - wait - move

#     print("\n================ ACTION DISTRIBUTION ================")
#     print(f"Total actions: {total}")
#     print(f"WAIT (.): {wait} ({wait/total:.2%})")
#     print(f"MOVE (hjkl+yubn): {move} ({move/total:.2%})")
#     print(f"OTHER: {other} ({other/total:.2%})")
#     print("=====================================================\n")

#     return results


# # if __name__ == "__main__":
# #     from agent.local_llm_client import LocalLLMClient

# #     llm_client = LocalLLMClient(
# #         model_path="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
# #         temperature=0.0,
# #         max_tokens=16,
# #     )

# #     print("Running frozen LLM baseline")
# #     print("-" * 40)

# #     run_llm_frozen(
# #         seed=42,
# #         episodes=10,
# #         llm_client=llm_client,
# #         prompt_path="prompts/v0_frozen.txt",
# #     )
