from experiments.nle_experiment_runner import NLEExperimentRunner
from agent.nethack_agent import NetHackAgent

SEEDS = [42, 123, 456, 789, 999, 2023, 2024, 31415, 27182, 16180]
MAX_STEPS = 2000

# Assume meta-learner already selected this prompt
agent = NetHackAgent(prompt_version="v1.2.0")

runner = NLEExperimentRunner(
    agent=agent,
    max_steps=MAX_STEPS,
)

runner.run(seeds=SEEDS, tag="proposed")
