import json
from pathlib import Path

from svr_new import svr_pipeline, save_result

HERE = Path(__file__).parent
CONFIGS_JSON = HERE / "configs.json"


def main():

    text = CONFIGS_JSON.read_text()
    configs = json.loads(text)

    print(f"{len(configs)} configurations to evaluate\n")

    for config in configs:

        result = svr_pipeline(**config)
        save_result(result, "llm")

        print(f"C={config['C']}, epsilon={config['epsilon']}, gamma={config['gamma']}"
              f" -> MAE {result['mae']:.4f}, R2 {result['r2']:.4f}")


if __name__ == "__main__":
    main()

