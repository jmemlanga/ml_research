"""bayesian_llm_init.py — LLM warm-start for optuna GPSampler + random-init baseline.

Runs narrow (shared box) and wide (shared box), 5 seeds each, both methods.
Does NOT clear the CSV — clear it once manually before running any baseline.
Needs llm_init_narrow.json and llm_init_wide.json (10 points each) alongside it.
"""

import os
import json

import optuna

from svr_new import svr_pipeline, save_result
from search_space import SEARCH_SPACE, SEARCH_SPACE_WIDE


def _to_tuples(space_dict):
    return {
        name: (d["low"], d["high"], d["scale"] == "log")
        for name, d in space_dict.items()
    }

SEARCH_SPACES = {
    "narrow": _to_tuples(SEARCH_SPACE),
    "wide":   _to_tuples(SEARCH_SPACE_WIDE),
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_STARTUP = 10
BUDGET    = 27
SEEDS     = [1, 2, 3, 4, 5]

_HERE = os.path.dirname(os.path.abspath(__file__))


def make_objective(space, space_name, method, seed):
    def objective(trial):
        C       = trial.suggest_float("C",       space["C"][0],       space["C"][1],       log=space["C"][2])
        gamma   = trial.suggest_float("gamma",   space["gamma"][0],   space["gamma"][1],   log=space["gamma"][2])
        epsilon = trial.suggest_float("epsilon", space["epsilon"][0], space["epsilon"][1], log=space["epsilon"][2])

        metrics = svr_pipeline(C=C, gamma=gamma, epsilon=epsilon)

        save_result(
            result=metrics,
            method=method, space=space_name, seed=seed, eval_index=trial.number,
        )
        return metrics["mae"]
    return objective


def run_bayesian(space_name, seed, llm_points=None):
    space  = SEARCH_SPACES[space_name]
    method = "bayesian_llm_init" if llm_points else "bayesian"

    sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=N_STARTUP)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    if llm_points:
        for p in llm_points:
            study.enqueue_trial(p)

    study.optimize(make_objective(space, space_name, method, seed), n_trials=BUDGET)
    return study


def run_all():
    for space_name in ["narrow", "wide"]:
        with open(os.path.join(_HERE, f"llm_init_{space_name}.json")) as fh:
            llm = json.load(fh)
        for seed in SEEDS:
            run_bayesian(space_name, seed, llm_points=None)
            run_bayesian(space_name, seed, llm_points=llm)


if __name__ == "__main__":
    run_all()