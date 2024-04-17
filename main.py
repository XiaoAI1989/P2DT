import logging

import hydra
import torch

import utils
from core import train, final_eval

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = utils.get_log_dict()
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, None, hydra.utils.get_original_cwd())
        final_eval(cfg.train.env_name, cfg.rtg_target, logger, cfg, seed, hydra.utils.get_original_cwd())


if __name__ == "__main__":
    main()
