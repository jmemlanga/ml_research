"""Shared hyperparameter search space.

Every search method (LLM, random, grid, Bayesian) must sample from these
exact ranges, or the comparison between them is meaningless.

Bounds chosen to match the region the LLM runs actually explored, with no
narrowing based on what those runs found. Narrowing would hand the
baselines knowledge the LLMs had to spend evaluations to acquire.

scale is "log" or "linear" and controls how a sampler should draw values.
"""

SEARCH_SPACE = {
    "C":       {"low": 1.0,   "high": 50000.0, "scale": "log"},
    "gamma":   {"low": 1e-5,  "high": 0.05,    "scale": "log"},
    "epsilon": {"low": 0.0,   "high": 1.0,     "scale": "linear"},
}