from pathlib import Path

from omegaconf import DictConfig, OmegaConf

def get_config() -> DictConfig:
    return OmegaConf.load(
        Path(__file__).resolve().parent.parent / "omegaconf.yaml"
    )
