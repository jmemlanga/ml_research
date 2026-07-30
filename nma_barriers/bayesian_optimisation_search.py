'''Bayesian optimisation baseline using optuna's GPSampler. 

Same pipeline, same folds, same results format as the other search methods, 
so the only thing that differs is how configurations are proposed.

'''

import optuna 
from optuna.samplers import GPSampler 

from search_space import SEARCH_SPACE
from svr_new import svr_pipeline, save_result

SEARCH_SEED = 1
N_ITER = 10
N_STARTUP = 0 #Number of configurations used before the GP takes over 

METHOD = "bayesian"

def objective(trial):
    config = {}
    for name, spec in SEARCH_SPACE.items():
        config[name] = trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            log=(spec["scale"]== "log")
        )

    result = svr_pipeline(**config)

    save_result(
        result,
        method=METHOD,
        seed=SEARCH_SEED,
        eval_index=trial.number + 1,
    )

    return result["mae"]

def main(): 
    sampler = GPSampler(
        seed=SEARCH_SEED,
        n_startup_trials=N_STARTUP,
        deterministic_objective=True,

    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",sampler=sampler)
    study.optimize(objective, n_trials=N_ITER)
    print(f"best MAE:{study.best_value:.4f}")
    print(f"best config:{study.best_params}")

if __name__ == "__main__":
    main()
