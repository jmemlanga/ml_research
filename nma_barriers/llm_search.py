"""LLM one-shot baseline — evaluates the LLM's proposed configs, narrow + wide.

Reads configs.json (narrow) and configs_wide.json (wide). Each should hold the
LLM's one-shot picks for that box. seed=0 (deterministic — one fixed config set
per box). Distinct from bayesian_llm_init, which GP-warmstarts from LLM points.
"""

import json
from pathlib import Path

from svr_new import svr_pipeline, save_result

HERE = Path(__file__).parent
CONFIG_FILES = {
    "narrow": HERE / "configs.json",
    "wide":   HERE / "configs_wide.json",
}


def main():
    for space_name, path in CONFIG_FILES.items():
        configs = json.loads(path.read_text())
        for i, config in enumerate(configs):
            result = svr_pipeline(**config)
            save_result(result, method="llm", space=space_name, seed=0, eval_index=i)


if __name__ == "__main__":
    main()
