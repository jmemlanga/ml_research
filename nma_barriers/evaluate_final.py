"""evaluate_final.py — each method's best training config, evaluated on test + lit.
Run AFTER all training. Uses the narrow (shared, fair) box for the comparison."""

import pandas as pd

from svr_new import RESULTS_CSV, svr_test, svr_lit_test, save_result

EVAL_SPACE = "narrow"


def best_configs():
    df = pd.read_csv(RESULTS_CSV)
    df = df[(df["method"] != "cv") & (df["space"] == EVAL_SPACE)]
    df = df[pd.to_numeric(df["eval_index"], errors="coerce").notna()]   # training rows only
    df = df.dropna(subset=["mae"])
    idx = df.groupby("method")["mae"].idxmin()
    return df.loc[idx, ["method", "C", "epsilon", "gamma", "mae"]]


def run():
    best = best_configs()
    print(f"Best training config per method ({EVAL_SPACE} box):\n")
    for _, r in best.iterrows():
        cfg = dict(C=r["C"], epsilon=r["epsilon"], gamma=r["gamma"])

        test_res = svr_test(**cfg)
        save_result(test_res, method=r["method"], space=EVAL_SPACE, seed=None, eval_index="TEST")

        lit_res = svr_lit_test(**cfg)
        save_result(lit_res, method=r["method"], space=EVAL_SPACE, seed=None, eval_index="LIT")

        print(f"{r['method']:<20} train {r['mae']:.3f} | test {test_res['mae']:.3f} | lit {lit_res['mae']:.3f}")


if __name__ == "__main__":
    run()