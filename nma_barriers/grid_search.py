"""Grid search baseline — narrow box only, 3x3x3 = 27 evals, budget-matched.

Deterministic (no seeds, seed=0 fixed). A grid over the wide box would be
meaningless (3 points over 9 decades of C), so grid runs narrow only.
eval_index is the grid-point number so it shares the plot's x-axis.
"""

import itertools
import numpy as np

from search_space import SEARCH_SPACE
from svr_new import svr_pipeline, save_result


def _axis(spec, n):
    if spec["scale"] == "log":
        return np.logspace(np.log10(spec["low"]), np.log10(spec["high"]), n)
    return np.linspace(spec["low"], spec["high"], n)


def main():
    C_grid   = _axis(SEARCH_SPACE["C"], 3)
    g_grid   = _axis(SEARCH_SPACE["gamma"], 3)
    eps_grid = _axis(SEARCH_SPACE["epsilon"], 3)

    for i, (c, eps, g) in enumerate(itertools.product(C_grid, eps_grid, g_grid)):
        result = svr_pipeline(C=float(c), epsilon=float(eps), gamma=float(g))
        save_result(result, method="grid", space="narrow", seed=0, eval_index=i)


if __name__ == "__main__":
    main()