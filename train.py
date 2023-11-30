import os
import logging


import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)

def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model


    Args:
        cfg: hydra config

    """

@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    logging.info(OmegaConf.to_yaml(cfg))

if __name__ == '__main__':
    run_model()