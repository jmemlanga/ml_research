"""
llm_hpo.py — automated LLM-driven hyperparameter optimisation for the Chemprop
solubility D-MPNN.

The LLM proposes one config at a time; we evaluate it with the shared
`train_and_evaluate` objective and feed the (config -> val RMSE) history back in
the next prompt. This is the closed-loop SVR experiment with the human replaced
by code.

Prerequisites:
  - pip install google-genai   (into the `solubility` env)
  - export GEMINI_API_KEY=...   (never hard-code it — shared repo)
  - train_chemprop.py must return "val_rmse" from train_and_evaluate
    (the monitor="val/rmse" change we made).
"""

import os
import json
import time
from pathlib import Path

from google import genai
from google.genai import types

from train_chemprop import Config, load_split, build_datasets, train_and_evaluate


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
MODEL       = "gemini-3.5-flash"   # set to the current free-tier Flash model
N_TRIALS    = 10                   # number of REAL evaluations (invalid proposals don't count)
MAX_RETRIES = 4                    # per trial: how many times we'll re-ask after an invalid config
SEED        = 0                    # fixed across all trials -> we compare configs, not seed luck
SPACE_NAME  = "wide"
RESULTS_CSV = Path(__file__).parent / "data" / "results" / "chemprop_hpo.csv"


# ---------------------------------------------------------------------------
# Search space  (keys MUST match Config field names so Config(**cfg) works)
#   ("int",   low, high)   -> integer in [low, high]
#   ("float", low, high)   -> float in   [low, high]
#   ("cat",   [options])   -> one of the listed strings
# ---------------------------------------------------------------------------
SPACE = {
    "depth":              ("int",   2,    8),
    "message_hidden_dim": ("int",   64,   1200),
    "ffn_hidden_dim":     ("int",   64,   1200),
    "ffn_num_layers":     ("int",   1,    4),
    "dropout":            ("float", 0.0,  0.6),
    "aggregation":        ("cat",   ["mean", "sum", "norm"]),
    "batch_size":         ("int",   16,   256),
    "max_lr":             ("float", 1e-4, 5e-2),   # spans orders of magnitude -> log-scale reasoning
    "max_epochs":         ("int",   20,   50),     # drives wall-clock; widen with care
}


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ---------------------------------------------------------------------------
# Turn SPACE into a JSON schema so Gemini returns valid, typed JSON.
# This enforces STRUCTURE (right keys, right JSON types). It does NOT enforce
# ranges — that's what validate() is for.
# ---------------------------------------------------------------------------
def build_schema(space):
    props = {}
    for key, spec in space.items():
        if spec[0] == "int":
            props[key] = {"type": "integer"}
        elif spec[0] == "float":
            props[key] = {"type": "number"}
        elif spec[0] == "cat":
            props[key] = {"type": "string", "enum": spec[1]}
    return {
        "type": "object",
        "properties": props,
        "required": list(space.keys()),
    }


# ---------------------------------------------------------------------------
# Build the prompt. This IS the experiment — so it contains ONLY what a fair
# optimiser is entitled to: the task, the search space, and the history so far.
# It deliberately contains NO domain hints, no known-good configs, no noise
# floor. Anything you wouldn't also hand the Bayesian arm would break the
# comparison.
# ---------------------------------------------------------------------------
def describe_space(space):
    lines = []
    for key, spec in space.items():
        if spec[0] == "int":
            lines.append(f"  - {key}: integer, {spec[1]} to {spec[2]}")
        elif spec[0] == "float":
            note = "  (spans orders of magnitude — reason on a log scale)" if key == "max_lr" else ""
            lines.append(f"  - {key}: number, {spec[1]:g} to {spec[2]:g}{note}")
        elif spec[0] == "cat":
            lines.append(f"  - {key}: one of {spec[1]}")
    return "\n".join(lines)


def describe_history(history):
    if not history:
        return "No configurations have been tried yet."
    lines = []
    for i, entry in enumerate(history):
        lines.append(f"  trial {i}: {json.dumps(entry['config'])} -> val RMSE {entry['val_rmse']:.4f}")
    return "\n".join(lines)


def build_prompt(space, history, last_error=None):
    prompt = f"""You are choosing hyperparameters for a Chemprop directed message-passing \
neural network (D-MPNN) that predicts aqueous solubility (logS) of molecules.

Your objective: propose a configuration that MINIMISES validation RMSE. Lower is better.

Search space (propose exactly these fields, within these bounds):
{describe_space(space)}

Results so far:
{describe_history(history)}

Based on the results so far, propose the next single configuration to evaluate. \
Balance exploring unfamiliar regions against refining the most promising ones."""

    if last_error is not None:
        prompt += f"\n\nYour previous proposal was rejected: {last_error}\nPropose a corrected configuration."

    return prompt


# ---------------------------------------------------------------------------
# Validate a proposed config against the space: presence, type, range.
# Returns (ok: bool, error: str). The error string is fed back to the LLM.
# The schema already guarantees types, but we check defensively — the second
# gate catches in-range violations (e.g. depth=15) the schema can't express.
# ---------------------------------------------------------------------------
def validate(cfg, space):
    for key, spec in space.items():
        if key not in cfg:
            return False, f"missing required field '{key}'"
        v = cfg[key]

        if spec[0] == "int":
            # bool is a subclass of int in Python — exclude it explicitly
            if isinstance(v, bool) or not isinstance(v, int):
                return False, f"'{key}' must be an integer, got {v!r}"
            if not (spec[1] <= v <= spec[2]):
                return False, f"'{key}'={v} is outside [{spec[1]}, {spec[2]}]"

        elif spec[0] == "float":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, f"'{key}' must be a number, got {v!r}"
            if not (spec[1] <= v <= spec[2]):
                return False, f"'{key}'={v} is outside [{spec[1]}, {spec[2]}]"

        elif spec[0] == "cat":
            if v not in spec[1]:
                return False, f"'{key}'={v!r} must be one of {spec[1]}"

    return True, ""


# ---------------------------------------------------------------------------
# One call to Gemini. Retries transient API errors (rate limits, blips) with
# exponential backoff. Returns the parsed JSON dict, or {} if parsing fails
# (which validate() will then reject, triggering a re-ask).
# ---------------------------------------------------------------------------
def call_gemini(prompt, space, max_api_retries=3):
    for attempt in range(max_api_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=build_schema(space),
                    temperature=1.0,   # exploration; note: not reproducible run-to-run
                ),
            )
            return json.loads(resp.text)
        except json.JSONDecodeError:
            return {}   # malformed JSON -> let validate() reject and re-ask
        except Exception as e:
            wait = 2 ** attempt
            print(f"  API error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Gemini API failed after repeated retries")


# ---------------------------------------------------------------------------
# Append one evaluated trial to the results CSV (mirrors save_result in svr_new).
# ---------------------------------------------------------------------------
def log_trial(trial, cfg, val_rmse):
    import pandas as pd
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "method": "llm",
        "space": SPACE_NAME,
        "seed": SEED,
        "trial": trial,
        "val_rmse": val_rmse,
        **cfg,
    }])
    row.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------
def main():
    datasets = build_datasets(load_split(seed=42))   # featurise ONCE, reuse every trial

    history = []
    rejections = 0

    for trial in range(N_TRIALS):
        # --- get ONE valid config (invalid proposals re-ask, don't burn the budget) ---
        last_error = None
        cfg = None
        for attempt in range(MAX_RETRIES):
            proposal = call_gemini(build_prompt(SPACE, history, last_error), SPACE)
            ok, err = validate(proposal, SPACE)
            if ok:
                cfg = proposal
                break
            rejections += 1
            print(f"  rejected (attempt {attempt + 1}): {err}")
            last_error = err
        if cfg is None:
            raise RuntimeError(f"No valid config after {MAX_RETRIES} attempts on trial {trial}")

        # --- evaluate on the shared objective (only known Config fields passed) ---
        config = Config(**{k: cfg[k] for k in SPACE})
        result = train_and_evaluate(config, datasets, seed=SEED)
        val = result["val_rmse"]

        history.append({"config": cfg, "val_rmse": val})
        log_trial(trial, cfg, val)
        print(f"trial {trial}: val RMSE {val:.4f} | {cfg}")

        time.sleep(2)   # courtesy gap; trials are minutes apart anyway

    best = min(history, key=lambda h: h["val_rmse"])
    print(f"\nbest: val RMSE {best['val_rmse']:.4f} | {best['config']}")
    print(f"total invalid proposals rejected: {rejections}")


if __name__ == "__main__":
    main()