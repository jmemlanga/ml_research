"""Random search baseline.

Samples configurations from SEARCH_SPACE and evaluates each one through
svr_pipeline, the same function the LLM runs used. Same folds, same
RANDOM_STATE, same CSV format, so the only difference between this and
llm_search.py is who proposes the configurations.
"""

import numpy as np
from scipy.stats import loguniform, uniform

from search_space import SEARCH_SPACE
from svr_new import svr_pipeline, save_result

# Seed for the sampling of configurations. Deliberately separate from
# RANDOM_STATE in svr_new, which seeds the CV fold split. The folds must
# stay identical across every method; the draws must not.
SEARCH_SEED = 5
N_ITER = 30


def build_distributions(space):
    """Turn the search space spec into scipy distribution objects."""
    distributions = {}
    for name, spec in space.items():
        if spec["scale"] == "log":
            distributions[name] = loguniform(spec["low"], spec["high"])
        else:
            distributions[name] = uniform(
                loc=spec["low"],
                scale=spec["high"] - spec["low"],
            )
    return distributions


def sample_config(distributions, rng):
    """Draw one configuration from the shared random stream.

    The stream is passed in rather than re-seeded on every draw. A stream
    keeps its position, so consecutive draws differ, and a change of
    SEARCH_SEED produces a genuinely independent set of configurations.
    """
    return {
        name: float(dist.rvs(random_state=rng))
        for name, dist in distributions.items()
    }


def main():
    distributions = build_distributions(SEARCH_SPACE)
    rng = np.random.default_rng(SEARCH_SEED)

    best_mae = None

    for i in range(N_ITER):
        config = sample_config(distributions, rng)
        result = svr_pipeline(**config)
        save_result(result, f"random_search_seed{SEARCH_SEED}")

        if best_mae is None or result["mae"] < best_mae:
            best_mae = result["mae"]

        print(
            f"{i + 1:3d}  "
            f"C={config['C']:>12.4f}  "
            f"epsilon={config['epsilon']:.4f}  "
            f"gamma={config['gamma']:.6f}  "
            f"MAE={result['mae']:.4f}  "
            f"best={best_mae:.4f}"
        )

    print()
    print(f"Seed {SEARCH_SEED}: best MAE after {N_ITER} evaluations: {best_mae:.4f}")


if __name__ == "__main__":
    main()