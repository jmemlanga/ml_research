"""
plot_llm_curve.py — HPO learning (sample-efficiency) curve for the LLM arm.

x-axis: trial number.  y-axis: validation RMSE.
  - faint dots : each trial's own val RMSE (what the LLM proposed)
  - bold step  : best val RMSE SO FAR (cumulative min) — the convergence curve

UNITS: val RMSE is on the SCALED validation targets (val is normalised in
build_datasets). It's what HPO ranks on, but it is NOT the same scale as test
RMSE / the mean-predictor baseline (those are real logS). So do NOT draw the
2.463 baseline or the 1.145 noise floor on this axis — different units.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # headless-safe on the Remote-SSH node (no display)
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = Path(__file__).parent / "data" / "results" / "chemprop_hpo.csv"
METHOD      = "llm"
SPACE       = "wide"           # match your run; drop the filter if only one space exists
OUT_PNG     = Path(__file__).parent / "data" / "results" / "llm_learning_curve.png"


def load_run(method, space):
    df = pd.read_csv(RESULTS_CSV)
    df = df[(df["method"] == method) & (df["space"] == space)].copy()
    df = df.sort_values("trial").reset_index(drop=True)
    df["best_so_far"] = df["val_rmse"].cummin()   # running minimum = best found so far
    return df


def main():
    df = load_run(METHOD, SPACE)
    if df.empty:
        raise SystemExit(f"No rows for method={METHOD}, space={SPACE} in {RESULTS_CSV}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # each trial's own score — the exploration
    ax.scatter(df["trial"], df["val_rmse"], s=30, color="0.6",
               zorder=2, label="per-trial val RMSE")

    # best-so-far — the sample-efficiency / convergence curve
    ax.step(df["trial"], df["best_so_far"], where="post", color="C0",
            linewidth=2, zorder=3, label="best so far")

    # mark the incumbent best
    best_i = df["val_rmse"].idxmin()
    ax.scatter(df.loc[best_i, "trial"], df.loc[best_i, "val_rmse"],
               s=90, color="C3", zorder=4,
               label=f"best = {df.loc[best_i, 'val_rmse']:.4f}")

    ax.set_xlabel("trial")
    ax.set_ylabel("validation RMSE (scaled)")
    ax.set_title(f"LLM-driven HPO — learning curve ({SPACE} space)")
    ax.set_xticks(df["trial"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()