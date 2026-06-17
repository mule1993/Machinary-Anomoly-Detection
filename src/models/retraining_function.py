"""import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.train import train_pipeline

# =================== RETRAINING CONFIGURATION =================
DATA_PATH = clean_set_key = "data/year=2026/month=06/clean_training_set.csv"
dynamic_scale_weight = 1
# ==============================================================


@hydra.main(version_base=None, config_path="../../config/", config_name="config")
def retraining(config: DictConfig):
    OmegaConf.set_readonly(config, False)

    # Overrides for retraining,
    # data balance weight hyeperparameters change not architectural hyperparameters
    config.data_path = DATA_PATH
    config.hyperparameters.scale_pos_weight = dynamic_scale_weight
    train_pipeline(config)


if __name__ == "__main__":
    retraining()
"""
