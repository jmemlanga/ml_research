#!/usr/bin/env python3
"""Best-so-far CV MAE vs eval index for the budget-10 experiment.
One panel per search space. Solid line = mean across seeds; shaded band =
min-max spread. Single-run methods (closed-loop LLM) get no band, matching
the main pipeline convention. Dashed line = GP takeover for bayesian.

Assumes the CSV has columns: method, space, seed, eval_index, mae
(as written by save_result). Point RESULTS at your fresh budget-10 file.
"""
import pandas as pd
import matplotlib.pyplot as plt

RESULTS   = "data/results/svr_results.csv"          # the fresh budget-10 file
KEEP      = ["llm_closed_loop", "random", "bayesian"]
COLORS    = {"llm_closed_loop": "green", "random": "purple", "bayesian": "blue"}
LABELS    = {"llm_closed_loop": "LLM (closed-loop)", "random": "random", "bayesian": "bayesian"}
N_STARTUP = 4                                        # bayesian random-init count -> GP-takeover marker

df = pd.read_csv(RESULTS)
df = df[df["method"].isin(KEEP) & (df["method"] != "cv")]

def best_curve(g):
    g = g.sort_values("eval_index")
    return g.assign(best=g["mae"].cummin())

spaces = [s for s in ["narrow", "wide"] if s in df["space"].unique()]
fig, axes = plt.subplots(1, len(spaces), figsize=(6 * len(spaces), 5), squeeze=False)

for ax, space in zip(axes[0], spaces):
    sub = df[df["space"] == space]
    for method in KEEP:
        m = sub[sub["method"] == method]
        if m.empty:
            continue
        curves = (m.groupby("seed", group_keys=False).apply(best_curve)
                    .groupby("eval_index")["best"])
        mean, lo, hi = curves.mean(), curves.min(), curves.max()
        ax.plot(mean.index, mean, color=COLORS[method], label=LABELS[method])
        if (hi - lo).abs().sum() > 0:                # skip band for single-run methods
            ax.fill_between(mean.index, lo, hi, color=COLORS[method], alpha=0.15)
    ax.axvline(N_STARTUP - 0.5, ls="--", color="grey", lw=1, alpha=0.7)   # GP takes over
    ax.set_title(f"{space} space")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best-so-far CV MAE")
    ax.legend()

fig.tight_layout()
out = "data/results/budget10_sample_efficiency.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")