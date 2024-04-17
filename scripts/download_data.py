"""Download D4RL gym trajectories and convert them to the pickle format used by buffer.py.

Pulls the episodic D4RL data re-hosted on the HuggingFace Hub
(edbeeching/decision_transformer_gym_replay, the dataset used by the official
Decision Transformer integration), so neither mujoco-py nor the original d4rl
package is required.

Usage (from the repository root):
    pip install datasets
    python scripts/download_data.py                 # halfcheetah/hopper/walker2d, medium
    python scripts/download_data.py --dataset expert --envs hopper

Output: dataset/<env>-<dataset>.pkl, a list of trajectories, each a dict with
float32 arrays 'observations' (T, state_dim), 'actions' (T, action_dim) and
'rewards' (T,) — exactly what SequenceBuffer expects.
"""
import argparse
import os
import pickle

import numpy as np

HF_REPO = "edbeeching/decision_transformer_gym_replay"
ALL_ENVS = ["halfcheetah", "hopper", "walker2d"]


def convert_split(env: str, dataset: str, out_dir: str) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "the 'datasets' package is required: pip install datasets"
        ) from exc

    hf_config = f"{env}-{dataset}-v2"
    print(f"downloading {HF_REPO}:{hf_config} ...")
    data = load_dataset(HF_REPO, hf_config, split="train")

    trajectories = []
    for episode in data:
        trajectories.append({
            "observations": np.asarray(episode["observations"], dtype=np.float32),
            "actions": np.asarray(episode["actions"], dtype=np.float32),
            "rewards": np.asarray(episode["rewards"], dtype=np.float32),
        })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env}-{dataset}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    steps = sum(len(t["rewards"]) for t in trajectories)
    print(f"wrote {out_path}: {len(trajectories)} trajectories, {steps} steps")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=ALL_ENVS, choices=ALL_ENVS)
    parser.add_argument("--dataset", default="medium", choices=["medium", "medium-replay", "expert"])
    parser.add_argument("--out-dir", default="dataset")
    args = parser.parse_args()

    for env in args.envs:
        convert_split(env, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
