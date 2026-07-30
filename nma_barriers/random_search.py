"""Random search baseline — 5 seeds, narrow + wide, shared schema, 27 evals."""

import numpy as np
from scipy.stats import loguniform, uniform

from search_space import SEARCH_SPACE , SEARCH_SPACE_WIDE
from svr_new import svr_pipeline, save_result

SEEDS  = [1]
N_ITER = 10
SPACES = {"wide": SEARCH_SPACE}#, "wide": SEARCH_SPACE_WIDE} excluded wide for this run now 


def build_distributions(space):
    d = {}
    for name, spec in space.items():
        if spec["scale"] == "log":
            d[name] = loguniform(spec["low"], spec["high"])
        else:
            d[name] = uniform(loc=spec["low"], scale=spec["high"] - spec["low"])
    return d


def main():
    for space_name, space in SPACES.items():
        dists = build_distributions(space)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for i in range(N_ITER):
                config = {name: float(dist.rvs(random_state=rng)) for name, dist in dists.items()}
                result = svr_pipeline(**config)
                save_result(result, method="random", space=space_name, seed=seed, eval_index=i)


if __name__ == "__main__":
    main()