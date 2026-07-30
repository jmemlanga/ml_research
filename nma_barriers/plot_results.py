"""plot_results.py — training sample-efficiency lines + test/lit bar charts."""

import os
import pandas as pd
import matplotlib.pyplot as plt

from svr_new import RESULTS_CSV

N_STARTUP = 10

METHOD_STYLE = {
    "random":            ("Random search",             "#9467bd"),
    "grid":              ("Grid search",               "#8c564b"),
    "llm":               ("LLM (one-shot)",            "#2ca02c"),
    "bayesian":          ("Bayesian (random init)",    "#1f77b4"),
    "bayesian_llm_init": ("Bayesian (LLM warm-start)", "#d62728"),
}


def _load():
    df = pd.read_csv(RESULTS_CSV)
    return df[df["method"] != "cv"].dropna(subset=["method", "mae"])


def plot_training():
    df = _load()
    df = df[pd.to_numeric(df["eval_index"], errors="coerce").notna()].copy()
    df["eval_index"] = df["eval_index"].astype(int)
    df = df.sort_values(["method", "space", "seed", "eval_index"])
    df["best_so_far"] = df.groupby(["method", "space", "seed"])["mae"].cummin()

    spaces = [s for s in ["narrow", "wide"] if s in df["space"].unique()]
    fig, axes = plt.subplots(1, len(spaces), figsize=(7 * len(spaces), 5.2), squeeze=False)
    axes = axes[0]

    for ax, space_name in zip(axes, spaces):
        sdf = df[df["space"] == space_name]
        for method, (label, colour) in METHOD_STYLE.items():
            mdf = sdf[sdf["method"] == method]
            if mdf.empty:
                continue
            agg = mdf.groupby("eval_index")["best_so_far"].agg(["mean", "min", "max"])
            ax.plot(agg.index, agg["mean"], label=label, color=colour, linewidth=2.2, zorder=3)
            ax.fill_between(agg.index, agg["min"], agg["max"], color=colour, alpha=0.15, zorder=2)
        ax.axvline(N_STARTUP - 0.5, ls="--", color="0.6", lw=1.2)
        ax.set_title(f"{space_name.capitalize()} box", fontsize=13, weight="bold")
        ax.set_xlabel("Evaluation index")
        ax.set_ylabel("Best-so-far CV MAE (kcal/mol)")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("Training: sample efficiency (best-so-far CV MAE)", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])   # leave room at bottom for legend, top for title
    _save(fig, "training_sample_efficiency.png")


def plot_final(marker, title, fname):
    df = _load()
    sub = df[df["eval_index"].astype(str) == marker].copy()
    if sub.empty:
        print(f"no {marker} rows — run evaluate_final.py first")
        return
    order   = [m for m in METHOD_STYLE if m in sub["method"].values]
    maes    = [sub[sub["method"] == m]["mae"].iloc[0] for m in order]
    labels  = [METHOD_STYLE[m][0] for m in order]
    colours = [METHOD_STYLE[m][1] for m in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, maes, color=colours)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    _save(fig, fname)


def _save(fig, name):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    plot_training()
    plot_final("TEST", "Held-out test set (100 reactions)", "test_comparison.png")
    plot_final("LIT",  "Literature test set (37 reactions)", "lit_comparison.png")