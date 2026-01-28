import json
from glob import glob

from agent.episode_log import EpisodeLog
from evaluation.nle_metrics import NLEMetrics
from evaluation.bootstrap import bootstrap_ci
from evaluation.stat_tests import compare_methods


def load_logs(pattern):
    logs = []
    for path in glob(pattern):
        with open(path) as f:
            data = json.load(f)
            logs.append(EpisodeLog(**data))
    return logs


def evaluate(tag_a, tag_b):
    logs_a = load_logs(f"results_nle/{tag_a}_seed*.json")
    logs_b = load_logs(f"results_nle/{tag_b}_seed*.json")

    agg_a = NLEMetrics.aggregate(logs_a)
    agg_b = NLEMetrics.aggregate(logs_b)

    steps_a = [l.total_steps for l in logs_a]
    steps_b = [l.total_steps for l in logs_b]

    ci_a = bootstrap_ci(steps_a)
    ci_b = bootstrap_ci(steps_b)

    test = compare_methods(steps_a, steps_b)

    return {
        tag_a: {**agg_a, "steps_ci": ci_a},
        tag_b: {**agg_b, "steps_ci": ci_b},
        "comparison": test,
    }


if __name__ == "__main__":
    result = evaluate("static", "proposed")
    print(json.dumps(result, indent=2))
