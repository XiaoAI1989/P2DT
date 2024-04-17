"""Re-run the continual-learning forgetting test on a finished run.

Usage:
    python eval.py                 # evaluates the most recent run under ./runs
    python eval.py --run-dir runs/2024-01-15/10-36-48_   # evaluate a specific run
"""
import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

import utils
from core import final_eval

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')


def get_latest_run_folder(runs_dir: Path) -> Path:
    run_dirs = [path for path in runs_dir.glob("*/*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No Hydra run directory found under {runs_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Hydra run directory of the experiment (default: latest under ./runs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the checkpoints to evaluate (default: first seed in the run's config)")
    args = parser.parse_args()

    utils.config_logging("final_eval.log")

    root_dir = Path.cwd()
    run_dir = args.run_dir if args.run_dir is not None else get_latest_run_folder(root_dir / "runs")
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    seed = args.seed if args.seed is not None else cfg.seeds[0]

    logger.info(f"Evaluating run {run_dir} with seed {seed}")
    final_eval(cfg.train.env_name, cfg.rtg_target, logger, cfg, seed,
               cmd=str(root_dir), exp_path=str(run_dir))


if __name__ == "__main__":
    main()
