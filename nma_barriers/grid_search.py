"""
Grid search baseline for the NMA barrier SVR.

Deterministic counterpart to random_search.py: instead of drawing configs from
distributions, it evaluates every point on a fixed 3 x 3 x 3 grid (27 configs,
budget-matched to the 30-evaluation random search).

Results are tagged 'grid_search' so aggregate_random_search.py-style tooling can
pick them out of the shared results CSV.
"""

import itertools

import numpy as np

# Same helpers random_search.py uses. Adjust the import path if that script
# defines them locally rather than in a shared module.
from svr_utils import svr_pipeline, save_result


# Log-spaced on C and gamma because both span orders of magnitude and the model
# responds to their ratio, not their absolute difference. Linear on epsilon
# because it lives on a bounded, roughly linear scale.
C_GRID = np.logspace(0, 4, 3)        # 1, 100, 10_000
GAMMA_GRID = np.logspace(-5, -1, 3)  # 1e-5, 1e-3, 1e-1
EPSILON_GRID = [0.05, 0.3, 0.8]

TAG = "grid_search"


def build_grid():
    """Every combination of the three axes, as a list of config dicts."""
    combos = itertools.product(C_GRID, EPSILON_GRID, GAMMA_GRID)
    return [{"C": c, "epsilon": eps, "gamma": g} for c, eps, g in combos]


def main():
    grid = build_grid()
    print(f"Grid: {len(C_GRID)} x {len(EPSILON_GRID)} x {len(GAMMA_GRID)} "
          f"= {len(grid)} configurations\n")

    best_mae = None
    best_config = None

    for i, config in enumerate(grid, start=1):
        result = svr_pipeline(**config)
        save_result(result, TAG)

        mae = result["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_config = config

        print(f"{i:>3}  C={config['C']:>12.4f}  epsilon={config['epsilon']:.4f}  "
              f"gamma={config['gamma']:.6f}  MAE={mae:.4f}  best={best_mae:.4f}")

    print(f"\nGrid search: best MAE after {len(grid)} evaluations: {best_mae:.4f}")
    print(f"Best config: {best_config}")


if __name__ == "__main__":
    main()