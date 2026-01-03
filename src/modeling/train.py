"""
Code to train models
"""

from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.modeling.environment import run_env, get_params_num


class NEAT:
    def __init__(
        self,
        inputs,
        outputs
    ):
        self.inputs = inputs
        self.outputs = outputs

    def train(self):
        pass

    def predict(self, observation):
        if observation[0] > 0:
            return 0
        return 1


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    inputs, outputs = get_params_num()
    model = NEAT(inputs, outputs)
    run_env(model)


if __name__ == "__main__":
    main()
