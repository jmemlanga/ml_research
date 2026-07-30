import json
from pathlib import Path
from svr_new import svr_pipeline, save_result

SPACE = "narrow"                       # "narrow" or "wide"
HERE  = Path(__file__).parent
FILE  = HERE / ("configs_cl.json" if SPACE == "narrow" else "configs_cl_wide.json")

def main():
    configs = json.loads(FILE.read_text())         # growing list, newest last
    i = len(configs) - 1
    result = svr_pipeline(**configs[i])
    save_result(result, method="llm_closed_loop", space=SPACE, seed=0, eval_index=i)
    print(f"trial {i} | {configs[i]} -> CV MAE = {result['mae']:.4f}")

if __name__ == "__main__":
    main()